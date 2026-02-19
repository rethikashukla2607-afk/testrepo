import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CORE AI FUNCTIONS ---

def extract_and_clean(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    
    text = text.lower()

    # --- NEW: Synonym Mapping ---
    synonyms = {
        "regular expressions": "regex",
        "natural language processing": "nlp",
        "artificial intelligence": "ai",
        "scikit learn": "scikit",
        "machine learning": "ml",
        "web development": "fullstack"
    }
    
    for full_form, short_form in synonyms.items():
        text = text.replace(full_form, short_form)
    # ----------------------------

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_match_score(job_desc, resume_text):
    """Calculates Cosine Similarity score using TF-IDF."""
    content = [job_desc, resume_text]
    cv = TfidfVectorizer(stop_words='english')
    matrix = cv.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    return round(similarity_matrix[0][1] * 100, 2)

def get_skill_analysis(resume_text):
    """Performs Boolean matching against a skill checklist."""
    categories = {
        "Languages": ["python", "java", "php", "sql", "javascript", "c++"],
        "AI/ML": ["nlp", "spacy", "scikit", "tensorflow", "regex", "tfidf"],
        "Tools": ["apache", "mysql", "streamlit", "github", "vs code", "pymupdf"]
    }
    results = []
    for cat, skills in categories.items():
        for skill in skills:
            status = "Matched" if skill in resume_text.lower() else "Missing"
            results.append({"Category": cat, "Skill": skill.upper(), "Status": status})
    return pd.DataFrame(results)

def generate_wordcloud(text):
    """Generates a visual Word Cloud of resume keywords."""
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="AI Resume Leaderboard", layout="wide")
st.title("🏆 Autonomous Resume Ranking & Analysis System")

# Sidebar
with st.sidebar:
    st.header("Project Methodology")
    st.write("**Algorithm:** TF-IDF & Cosine Similarity")
    st.write("**Preprocessing:** Regex Cleaning")
    st.info("Upload multiple resumes to see the ranking leaderboard.")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📁 Input Data")
    jd_input = st.text_area("Paste Job Description:", height=200, placeholder="Enter the job requirements here...")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

with col2:
    st.subheader("📊 Analysis Results")
    
    if st.button("Rank & Analyze Resumes"):
        if jd_input and uploaded_files:
            batch_results = []
            
            # Use a progress bar for visual effect
            progress_bar = st.progress(0)
            
            for index, file in enumerate(uploaded_files):
                # Process each file
                resume_text = extract_and_clean(file)
                score = get_match_score(jd_input, resume_text)
                
                # Skill Analysis for this specific file
                df_skills = get_skill_analysis(resume_text)
                matched = df_skills[df_skills['Status'] == 'Matched']['Skill'].tolist()
                missing = df_skills[df_skills['Status'] == 'Missing']['Skill'].tolist()
                
                batch_results.append({
                    "Candidate": file.name,
                    "Match Score": score,
                    "Matched Skills": ", ".join(matched),
                    "Text": resume_text  # Saved for word cloud of the top candidate
                })
                
                progress_bar.progress((index + 1) / len(uploaded_files))
            
            # Create Leaderboard
            leaderboard_df = pd.DataFrame(batch_results).sort_values(by="Match Score", ascending=False)
            
            # 1. Display Table
            st.write("### Candidate Leaderboard")
            st.dataframe(leaderboard_df[["Candidate", "Match Score", "Matched Skills"]], use_container_width=True)
            
            # 2. Top Candidate Visualization
            top_candidate = leaderboard_df.iloc[0]
            st.divider()
            st.subheader(f"⭐ Analysis for Top Candidate: {top_candidate['Candidate']}")
            
            wc_col, list_col = st.columns(2)
            with wc_col:
                st.write("**Keyword Cloud:**")
                st.pyplot(generate_wordcloud(top_candidate['Text']))
            
            with list_col:
                st.write("**Missing Skills Gap:**")
                # Showing missing skills for the top candidate only
                full_skills = get_skill_analysis(top_candidate['Text'])
                missing_list = full_skills[full_skills['Status'] == 'Missing']['Skill'].tolist()
                for skill in missing_list:
                    st.write(f"❌ {skill}")

            # 3. Export
            csv = leaderboard_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Leaderboard CSV", csv, "leaderboard.csv", "text/csv")
            
        else:
            st.warning("Please provide a Job Description and at least one Resume PDF.")