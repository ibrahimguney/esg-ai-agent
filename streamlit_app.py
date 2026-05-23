import os
import json
import re
from io import BytesIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_openai import ChatOpenAI


# =========================
# ENV
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="ESG-AI Agent",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 ESG-AI Agent")
st.subheader("Sürdürülebilirlik Raporlarından ESG Veri Çıkarımı ve Greenwashing Risk Analizi")

st.markdown(
    """
Bu uygulama; şirket sürdürülebilirlik raporlarından ESG değişkenlerini çıkarır, 
kanıt cümleleri üretir, metinsel açıklama kalitesini skorlar ve sonuçları Excel olarak indirmenizi sağlar.
"""
)


# =========================
# HELPERS
# =========================

def read_pdf_text(uploaded_file):
    """PDF dosyasından sayfa bazlı metin çıkarır."""
    reader = PdfReader(uploaded_file)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        if text.strip():
            pages.append({
                "page_no": i,
                "text": text
            })

    return pages
def extract_json_from_response(text):
    """
    LLM yanıtındaki JSON bölümünü ayıklar.
    Model bazen JSON'u düz metin veya markdown içinde döndürebilir.
    """
    import json
    import re

    if not text:
        return None

    # Önce doğrudan JSON olarak dene
    try:
        return json.loads(text)
    except Exception:
        pass

    # ```json ... ``` bloğu varsa onu yakala
    code_block = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except Exception:
            pass

    # Genel JSON nesnesini yakala
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None

def limit_text_by_keywords(pages, keywords, max_chars=60000):
    """
    ESG anahtar kelimelerinin geçtiği sayfaları ve komşu sayfaları seçer.
    Böylece tablo bir önceki/sonraki sayfadaysa kaçırılmaz.
    """
    selected_indexes = set()

    for idx, page in enumerate(pages):
        text_lower = page["text"].lower()

        if any(k.lower() in text_lower for k in keywords):
            selected_indexes.add(idx)

            if idx > 0:
                selected_indexes.add(idx - 1)

            if idx < len(pages) - 1:
                selected_indexes.add(idx + 1)

    selected = []

    for idx in sorted(selected_indexes):
        page = pages[idx]
        selected.append(f"\n--- PAGE {page['page_no']} ---\n{page['text']}")

    combined = "\n".join(selected)

    if not combined.strip():
        combined = "\n".join(
            [f"\n--- PAGE {p['page_no']} ---\n{p['text']}" for p in pages[:20]]
        )

    return combined[:max_chars]

def validate_evidence_against_text(variables_df, report_text):
    """
    Evidence sentence birebir bulunmasa bile,
    değer ve anahtar kelime rapor metninde geçiyorsa kanıtı kabul eder.
    """
    if variables_df.empty:
        return variables_df

    report_text_clean = " ".join(report_text.split()).lower()

    for idx, row in variables_df.iterrows():
        evidence = str(row.get("evidence_sentence", "")).strip()
        value = str(row.get("value", "")).strip()
        variable = str(row.get("variable", "")).strip().lower()

        if not evidence or evidence.lower() in ["not disclosed", "none", "nan"]:
            variables_df.at[idx, "manual_check"] = "Yes"
            variables_df.at[idx, "confidence"] = 0
            variables_df.at[idx, "validation_note"] = "No evidence sentence"
            continue

        if value.lower() in ["not disclosed", "none", "nan", ""]:
            variables_df.at[idx, "manual_check"] = "Yes"
            variables_df.at[idx, "confidence"] = 0
            variables_df.at[idx, "validation_note"] = "No disclosed value"
            continue

        evidence_clean = " ".join(evidence.split()).lower()

        value_alt_1 = value.replace(".", "").replace(",", ".")
        value_alt_2 = value.replace(".", "")
        value_alt_3 = value.replace(",", ".")

        value_found = (
            value.lower() in report_text_clean
            or value_alt_1.lower() in report_text_clean
            or value_alt_2.lower() in report_text_clean
            or value_alt_3.lower() in report_text_clean
        )

        keyword_map = {
            "scope1_tco2e": ["kapsam 1", "sera gazı", "emisyon"],
            "scope2_tco2e": ["kapsam 2", "sera gazı", "emisyon"],
            "scope3_tco2e": ["kapsam 3", "sera gazı", "emisyon"],
            "total_ghg_tco2e": ["kapsam 1", "kapsam 2", "sera gazı"],
            "carbon_intensity": ["yoğunluğu", "sera gazı"],
            "energy_consumption": ["toplam enerji tüketimi"],
            "water_consumption": ["toplam su tüketimi"],
            "waste_generated": ["toplam atık miktarı"],
            "employee_number": ["toplam çalışan sayısı"],
            "female_employee_ratio": ["kadın", "%"],
            "female_manager_ratio": ["kadın yönetici oranı"],
            "training_hours": ["toplam eğitim"],
            "turnover_rate": ["çalışan devir oranı"],
            "female_board_ratio": ["yönetim kurulu", "kadın"]
        }

        expected_keywords = keyword_map.get(variable, [])
        keyword_found = any(k in report_text_clean for k in expected_keywords)

        exact_evidence_found = evidence_clean in report_text_clean

        if exact_evidence_found or (value_found and keyword_found):
            variables_df.at[idx, "manual_check"] = "No"
            variables_df.at[idx, "confidence"] = 90
            variables_df.at[idx, "validation_note"] = "Evidence/value found in selected report text"
        else:
            variables_df.at[idx, "manual_check"] = "Yes"
            variables_df.at[idx, "confidence"] = 0
            variables_df.at[idx, "validation_note"] = "Evidence sentence not found in selected report text"

    return variables_df

def get_llm():
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY bulunamadı. .env dosyasına OPENAI_API_KEY ekleyiniz.")
        st.stop()

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2
    )
def select_text_by_page_range(pages, start_page, end_page, max_chars=60000):
    """
    Belirli sayfa aralığındaki metni seçer.
    Özellikle uzun sürdürülebilirlik raporlarında performans tablolarını yakalamak için kullanılır.
    """
    selected = []

    for page in pages:
        if start_page <= page["page_no"] <= end_page:
            selected.append(f"\n--- PAGE {page['page_no']} ---\n{page['text']}")

    combined = "\n".join(selected)

    if not combined.strip():
        return ""

    return combined[:max_chars]

def create_extraction_prompt(company, year, report_text):
    return f"""
Sen bir ESG veri çıkarım ve sürdürülebilirlik raporu analiz agentısın.

Görevin, verilen sürdürülebilirlik / entegre faaliyet / TSRS raporu metninden ESG değişkenlerini çıkarmaktır.

Kurallar:
1. Sadece metinde açıkça verilen bilgileri çıkar.
2. Tahmin yapma.
3. Değer yoksa "Not disclosed" yaz.
4. Her değer için kanıt cümlesi ver.
5. Her değer için sayfa numarası varsa belirt.
6. Birim, yıl ve güven skorunu ayrıca yaz.
7. Greenwashing için kesin hüküm verme; yalnızca risk skoru üret.
8. Çıktıyı SADECE geçerli JSON olarak ver. Markdown kullanma.
9. Türkçe raporlarda Scope 1 yerine "Kapsam 1", Scope 2 yerine "Kapsam 2", Scope 3 yerine "Kapsam 3" ifadeleri geçebilir.
10. "sera gazı emisyonları", "karbon emisyonları", "CO₂e", "tCO₂e" ifadelerini emisyon değişkenleri için dikkate al.
11. Tablolarda verilen sayısal değerleri de çıkar.
12. Bir değer tabloda varsa ve cümle yoksa evidence_sentence alanına tablo satırının açıklamasını yaz.
13. Sayı Türkçe formatta verilmişse, örneğin 6.172.417, bunu aynen koru.
14. Eğer değer rapor metninde açıkça yoksa kesinlikle örnek veya tahmini değer üretme.
15. "olarak belirtilmiştir" gibi genel kanıt cümlesi yazma; evidence_sentence alanı rapor metninden aynen alınmış gerçek cümle veya tablo satırı olmalıdır.
16. evidence_sentence alanındaki metin, verilen rapor metninde birebir bulunmalıdır.
17. Eğer evidence_sentence rapor metninden birebir alınamıyorsa value alanına "Not disclosed", confidence alanına 0, manual_check alanına "Yes" yaz.
18. confidence alanı 0-100 aralığında olmalıdır. Açıkça bulunan değerlerde 80-100, bulunmayanlarda 0 yaz.
19. net_zero_target değişkeni için değer sadece "Yes", "No" veya "Not disclosed" olmalıdır. Hedef yılı ayrı olarak target_year değişkenine yaz.

Şirket: {company}
Yıl: {year}

Çıkarılacak değişkenler:

Çevresel:
- scope1_tco2e
- scope2_tco2e
- scope3_tco2e
- total_ghg_tco2e
- carbon_intensity
- energy_consumption
- renewable_energy_ratio
- water_consumption
- waste_generated
- net_zero_target
- target_year

Sosyal:
- employee_number
- female_employee_ratio
- female_manager_ratio
- training_hours
- ltifr
- accident_rate
- turnover_rate
- human_rights_policy

Yönetişim:
- board_independence_ratio
- female_board_ratio
- esg_committee
- ethics_policy
- anti_corruption_policy
- whistleblowing_mechanism
- esg_incentive

Metinsel skorlar:
- vague_statement_ratio
- quantitative_evidence_ratio
- evidence_backed_claim_ratio
- gri_score
- tcfd_score

JSON şeması:

{{
  "company": "{company}",
  "year": "{year}",
  "extracted_variables": [
    {{
      "variable": "",
      "value": "",
      "unit": "",
      "evidence_sentence": "",
      "page_no": "",
      "confidence": 0,
      "manual_check": "Yes/No"
    }}
  ],
  "textual_scores": {{
    "vague_statement_ratio": 0,
    "quantitative_evidence_ratio": 0,
    "evidence_backed_claim_ratio": 0,
    "gri_score": 0,
    "tcfd_score": 0
  }},
  "short_assessment": ""
}}

Rapor metni:
{report_text}
"""


def calculate_scores(textual_scores):
    vague = float(textual_scores.get("vague_statement_ratio", 0) or 0)
    quant = float(textual_scores.get("quantitative_evidence_ratio", 0) or 0)
    evidence = float(textual_scores.get("evidence_backed_claim_ratio", 0) or 0)
    gri = float(textual_scores.get("gri_score", 0) or 0)
    tcfd = float(textual_scores.get("tcfd_score", 0) or 0)

    text_index = (quant + evidence + (100 - vague)) / 3
    greenwashing_score = 0.40 * vague + 0.30 * (100 - quant) + 0.30 * (100 - evidence)

    if greenwashing_score <= 33:
        risk_level = "Düşük"
    elif greenwashing_score <= 66:
        risk_level = "Orta"
    else:
        risk_level = "Yüksek"

    return {
        "vague_statement_ratio": vague,
        "quantitative_evidence_ratio": quant,
        "evidence_backed_claim_ratio": evidence,
        "gri_score": gri,
        "tcfd_score": tcfd,
        "TEXTIndex": round(text_index, 2),
        "Greenwashing_Risk_Score": round(greenwashing_score, 2),
        "Risk_Level": risk_level
    }


def create_excel(company, year, variables_df, scores_df, assessment):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        variables_df.to_excel(writer, index=False, sheet_name="Kanitli_ESG_Verileri")
        scores_df.to_excel(writer, index=False, sheet_name="Metinsel_Skorlar")

        notes_df = pd.DataFrame({
            "Alan": [
                "Şirket",
                "Yıl",
                "Kısa Değerlendirme",
                "TEXTIndex Formülü",
                "Greenwashing Risk Formülü",
                "Not"
            ],
            "Açıklama": [
                company,
                year,
                assessment,
                "(QuantitativeEvidence + EvidenceBacked + (100 - Vague)) / 3",
                "0.40*Vague + 0.30*(100-QuantitativeEvidence) + 0.30*(100-EvidenceBacked)",
                "AI çıktıları akademik kullanım öncesinde rapor sayfası ve kanıt cümlesiyle manuel doğrulanmalıdır."
            ]
        })
        notes_df.to_excel(writer, index=False, sheet_name="Notlar")

    output.seek(0)
    return output
def select_text_by_page_range(pages, start_page, end_page, max_chars=60000):
    selected = []

    for page in pages:
        if start_page <= page["page_no"] <= end_page:
            selected.append(f"\n--- PAGE {page['page_no']} ---\n{page['text']}")

    combined = "\n".join(selected)
    return combined[:max_chars]


# =========================
# SIDEBAR
# =========================

st.sidebar.header("⚙️ Ayarlar")

company = st.sidebar.text_input("Şirket adı", value="Tüpraş")
year = st.sidebar.text_input("Rapor yılı", value="2024")

keywords_input = st.sidebar.text_area(
    "Aranacak anahtar kelimeler",
    value="Scope 1, Scope 2, Scope 3, emissions, enerji, energy, renewable, yenilenebilir, net zero, net sıfır, GRI, TCFD, TSRS, female, kadın, employee, çalışan, ethics, etik, board, yönetim"
)


max_chars = st.sidebar.slider(
    "LLM'e gönderilecek maksimum karakter",
    min_value=10000,
    max_value=60000,
    value=60000,
    step=5000
)

start_page = st.sidebar.number_input("Başlangıç sayfası", min_value=1, value=376)
end_page = st.sidebar.number_input("Bitiş sayfası", min_value=1, value=412)

# =========================
# MAIN
# =========================

uploaded_file = st.file_uploader(
    "Sürdürülebilirlik / Entegre Faaliyet / TSRS rapor PDF dosyasını yükleyin",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"PDF yüklendi: {uploaded_file.name}")

    with st.spinner("PDF metni okunuyor..."):
        pages = read_pdf_text(uploaded_file)

    st.info(f"Toplam metin çıkarılan sayfa sayısı: {len(pages)}")

    report_text = select_text_by_page_range(
        pages,
        start_page=start_page,
        end_page=end_page,
        max_chars=max_chars
    )

    with st.expander("LLM'e gönderilecek seçilmiş metni göster ve düzenle", expanded=False):
        edited_report_text = st.text_area(
            "Gerekirse burada metni düzenleyebilirsiniz. LLM bu metni kullanacaktır.",
            value=report_text,
            height=400
        )

    report_text = edited_report_text

    if st.button("🚀 ESG Analizini Başlat"):
        llm = get_llm()
        prompt = create_extraction_prompt(company, year, report_text)

        with st.spinner("ESG değişkenleri çıkarılıyor..."):
            response = llm.invoke(prompt)
            raw_text = response.content

        result = extract_json_from_response(raw_text)

        if result is None:
            st.error("LLM yanıtı JSON formatında çözümlenemedi. Ham yanıt aşağıdadır.")
            st.text(raw_text)
            st.stop()

        extracted_variables = result.get("extracted_variables", [])
        textual_scores = result.get("textual_scores", {})
        assessment = result.get("short_assessment", "")

        variables_df = pd.DataFrame(extracted_variables)
        variables_df = validate_evidence_against_text(variables_df, report_text)

        if not variables_df.empty:
            variables_df.insert(0, "company", company)
            variables_df.insert(1, "year", year)
        score_result = calculate_scores(textual_scores)
        scores_df = pd.DataFrame([{
            "company": company,
            "year": year,
            **score_result
        }])

        st.subheader("📊 Kanıtlı ESG Veri Çıkarımı")
        st.dataframe(variables_df, use_container_width=True)

        st.subheader("🧮 Metinsel Skorlar")
        st.dataframe(scores_df, use_container_width=True)

        st.subheader("📝 Kısa Değerlendirme")
        st.write(assessment)

        excel_file = create_excel(company, year, variables_df, scores_df, assessment)

        st.download_button(
            label="📥 Excel Olarak İndir",
            data=excel_file,
            file_name=f"{company}_{year}_ESG_AI_Agent_Cikarim.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.warning("Lütfen analiz için bir PDF raporu yükleyin.")