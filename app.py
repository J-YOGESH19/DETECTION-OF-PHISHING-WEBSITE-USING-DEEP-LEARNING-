

import streamlit as st
import re
import numpy as np
import pandas as pd
import joblib
import requests
import tldextract
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import tensorflow as tf
import time

# CONFIG
MODEL_PATH = "phishing_model.h5"
SCALER_PATH = "scaler.save"
FEATURE_COLUMNS = [
 "NumDots","SubdomainLevel","PathLevel","UrlLength","NumDash","NumDashInHostname","AtSymbol","TildeSymbol",
 "NumUnderscore","NumPercent","NumQueryComponents","NumAmpersand","NumHash","NumNumericChars","NoHttps",
 "RandomString","IpAddress","DomainInSubdomains","DomainInPaths","HttpsInHostname","HostnameLength","PathLength",
 "QueryLength","DoubleSlashInPath","NumSensitiveWords","EmbeddedBrandName","PctExtHyperlinks","PctExtResourceUrls",
 "ExtFavicon","InsecureForms","RelativeFormAction","ExtFormAction","AbnormalFormAction","PctNullSelfRedirectHyperlinks",
 "FrequentDomainNameMismatch","FakeLinkInStatusBar","RightClickDisabled","PopUpWindow","SubmitInfoToEmail","IframeOrFrame",
 "MissingTitle","ImagesOnlyInForm","SubdomainLevelRT","UrlLengthRT","PctExtResourceUrlsRT","AbnormalExtFormActionR",
 "ExtMetaScriptLinkRT","PctExtNullSelfRedirectHyperlinksRT"
]

SENSITIVE_WORDS = ["login","secure","account","update","verify","bank","submit","confirm","password","signin","paypal","wp-"]
BRAND_NAMES = ["google","facebook","paypal","amazon","microsoft","apple","bank"]

# Minimal Dark Theme CSS
st.markdown("""
<style>
    /* Dark background */
    .main {
        background-color: #0e1117;
    }
    
    .stApp {
        background-color: #0e1117;
    }
    
    /* Title styling */
    .title-text {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin: 2rem 0 1rem 0;
    }
    
    .subtitle-text {
        color: #a0a0a0;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Clean button */
    .stButton>button {
        background-color: #ffffff;
        color: #000000;
        padding: 0.6rem 2rem;
        border: none;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #e0e0e0;
    }
    
    /* Input field */
    .stTextInput>div>div>input {
        background-color: #1e1e1e;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 0.6rem;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #ffffff;
    }
    
    /* Result badges */
    .result-safe {
        background-color: #1e1e1e;
        border: 2px solid #00ff00;
        color: #00ff00;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0;
    }
    
    .result-danger {
        background-color: #1e1e1e;
        border: 2px solid #ff0000;
        color: #ff0000;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0;
    }
    
    .confidence-text {
        color: #a0a0a0;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Info cards */
    .info-card {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    
    .info-label {
        color: #a0a0a0;
        font-size: 0.9rem;
    }
    
    .info-value {
        color: #ffffff;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }
    
    /* Stats boxes */
    .stat-box {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        padding: 1.2rem;
        border-radius: 6px;
        text-align: center;
    }
    
    .stat-label {
        color: #a0a0a0;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    /* Section headers */
    h3 {
        color: #ffffff !important;
        font-weight: 500 !important;
        margin-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def is_ip(host):
    return bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host))

def count_digits(s):
    return sum(ch.isdigit() for ch in s)

def has_random_string(s):
    parts = re.findall(r'[A-Za-z0-9]{6,}', s)
    for p in parts:
        letters = sum(c.isalpha() for c in p)
        vowels = sum(c.lower() in 'aeiou' for c in p)
        if letters >= 6 and (vowels / (letters+1)) < 0.25:
            return 1
    return 0

def pct_external_links(soup, base_domain):
    links = soup.find_all('a', href=True)
    if not links:
        return 0.0
    ext = 0
    for a in links:
        href = a['href']
        parsed = urlparse(href)
        if parsed.netloc:
            if base_domain not in parsed.netloc:
                ext += 1
    return ext / len(links)

def pct_external_resources(soup, base_domain):
    resources = soup.find_all(['script','link','img'], src=True) + soup.find_all('link', href=True)
    if not resources:
        return 0.0
    ext = 0
    total = 0
    for tag in resources:
        href = tag.get('src') or tag.get('href') or ""
        if not href:
            continue
        total += 1
        parsed = urlparse(href)
        if parsed.netloc and base_domain not in parsed.netloc:
            ext += 1
    return ext / total if total else 0.0

def detect_right_click_disabled(soup):
    if soup.find(attrs={"oncontextmenu": True}):
        return 1
    scripts = soup.find_all('script')
    for s in scripts:
        text = (s.string or "")[:1000]
        if 'contextmenu' in text or 'oncontextmenu' in text:
            return 1
    return 0

def detect_iframe_frame(soup):
    if soup.find('iframe') or soup.find('frame'):
        return 1
    return 0

def missing_title(soup):
    if not soup.title or not soup.title.string or not soup.title.string.strip():
        return 1
    return 0

def extract_features_from_url(url):
    f = {c: 0 for c in FEATURE_COLUMNS}
    parsed = urlparse(url if url.startswith(("http://","https://")) else "http://" + url)
    scheme = parsed.scheme
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    netloc = parsed.netloc or ""
    full = url

    f["UrlLength"] = len(full)
    f["NumDots"] = full.count('.')
    f["NumDash"] = full.count('-')
    f["NumDashInHostname"] = host.count('-')
    f["AtSymbol"] = 1 if '@' in full else 0
    f["TildeSymbol"] = 1 if '~' in full else 0
    f["NumUnderscore"] = full.count('_')
    f["NumPercent"] = full.count('%')
    f["NumQueryComponents"] = len(parse_qs(query).keys()) if query else 0
    f["NumAmpersand"] = full.count('&')
    f["NumHash"] = full.count('#')
    f["NumNumericChars"] = count_digits(full)
    f["NoHttps"] = 1 if scheme != "https" else 0
    f["IpAddress"] = 1 if is_ip(host) else 0
    f["HostnameLength"] = len(host)
    f["PathLength"] = len(path)
    f["QueryLength"] = len(query)
    f["DoubleSlashInPath"] = 1 if '//' in path and path.index('//')>0 else 0

    ext = tldextract.extract(full)
    subdomain = ext.subdomain or ""
    if subdomain == "":
        f["SubdomainLevel"] = 0
    else:
        f["SubdomainLevel"] = subdomain.count('.') + 1

    f["PathLevel"] = len([p for p in path.split('/') if p])
    f["RandomString"] = has_random_string(host + path + query)

    domain = ext.domain or ""
    f["DomainInSubdomains"] = 1 if domain and domain in subdomain else 0
    f["DomainInPaths"] = 1 if domain and domain in path else 0
    f["HttpsInHostname"] = 1 if 'https' in host else 0

    lw = full.lower()
    f["NumSensitiveWords"] = sum(1 for w in SENSITIVE_WORDS if w in lw)
    f["EmbeddedBrandName"] = 1 if any(b in lw for b in BRAND_NAMES) else 0

    f["SubdomainLevelRT"] = f["SubdomainLevel"]
    f["UrlLengthRT"] = f["UrlLength"]

    html_features = {
        "PctExtHyperlinks": 0.0, "PctExtResourceUrls": 0.0, "ExtFavicon": 0,
        "InsecureForms": 0, "RelativeFormAction": 0, "ExtFormAction": 0, "AbnormalFormAction": 0,
        "PctNullSelfRedirectHyperlinks": 0.0, "FrequentDomainNameMismatch": 0, "FakeLinkInStatusBar": 0,
        "RightClickDisabled": 0, "PopUpWindow": 0, "SubmitInfoToEmail": 0, "IframeOrFrame": 0,
        "MissingTitle": 0, "ImagesOnlyInForm": 0, "PctExtResourceUrlsRT": 0.0,
        "AbnormalExtFormActionR": 0, "ExtMetaScriptLinkRT": 0.0, "PctExtNullSelfRedirectHyperlinksRT": 0.0
    }

    try:
        headers = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64)"}
        resp = requests.get(url if url.startswith(("http://","https://")) else "http://" + url, headers=headers, timeout=6)
        content_type = resp.headers.get('Content-Type','')
        if resp.status_code == 200 and 'text/html' in content_type:
            soup = BeautifulSoup(resp.text, "html.parser")
            base_domain = ext.registered_domain or domain

            html_features["PctExtHyperlinks"] = pct_external_links(soup, base_domain)
            html_features["PctExtResourceUrls"] = pct_external_resources(soup, base_domain)

            favicon = soup.find('link', rel=lambda v: v and 'icon' in v.lower())
            if favicon and favicon.has_attr('href'):
                fav_url = urlparse(favicon['href'])
                if fav_url.netloc and base_domain not in fav_url.netloc:
                    html_features["ExtFavicon"] = 1

            forms = soup.find_all('form')
            if forms:
                ext_action_count = 0
                null_self_redirects = 0
                insecure_form = 0
                for form in forms:
                    action = (form.get('action') or "").strip()
                    if action == "" or action == None:
                        null_self_redirects += 1
                    parsed_a = urlparse(action)
                    if parsed_a.netloc and base_domain not in parsed_a.netloc:
                        ext_action_count += 1
                    method = (form.get('method') or "").lower()
                    if method == 'get':
                        insecure_form = 1
                total_forms = len(forms)
                html_features["ExtFormAction"] = 1 if ext_action_count > 0 else 0
                html_features["InsecureForms"] = insecure_form
                html_features["PctNullSelfRedirectHyperlinks"] = null_self_redirects / total_forms if total_forms else 0.0

            html_features["IframeOrFrame"] = detect_iframe_frame(soup)
            html_features["RightClickDisabled"] = detect_right_click_disabled(soup)
            html_features["MissingTitle"] = missing_title(soup)

            if soup.find('a', onclick=True):
                html_features["FakeLinkInStatusBar"] = 1

            scripts = soup.find_all('script')
            for s in scripts:
                txt = s.string or ""
                if 'window.open' in txt or 'popup' in txt:
                    html_features["PopUpWindow"] = 1
                if 'mailto:' in txt or 'submit' in txt:
                    html_features["SubmitInfoToEmail"] = 1

    except Exception as e:
        pass

    for k,v in html_features.items():
        if k in f:
            f[k] = v

    features_ordered = [float(f.get(col, 0)) for col in FEATURE_COLUMNS]
    return features_ordered, f

# Main UI
st.markdown('<h1 class="title-text">Phishing URL Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Deep Learning-Based Malicious Website Detection</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load model
try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input
st.markdown("### Enter URL")
url_input = st.text_input("", placeholder="https://example.com", label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Analyze URL"):
    if not url_input.strip():
        st.warning("Please enter a URL")
    else:
        with st.spinner("Analyzing..."):
            features_list, features_dict = extract_features_from_url(url_input.strip())
            
            X = np.array([features_list], dtype=float)
            try:
                X_scaled = scaler.transform(X)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
            
            proba = model.predict(X_scaled)[0][0]
            label = int(proba > 0.5)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Results
        if label == 1:
            st.markdown(f"""
            <div class="result-danger">
                PHISHING DETECTED
            </div>
            <div class="confidence-text">Confidence: {proba*100:.1f}%</div>
            """, unsafe_allow_html=True)
            st.error("Warning: This URL exhibits characteristics of a phishing website")
        else:
            st.markdown(f"""
            <div class="result-safe">
                URL IS SAFE
            </div>
            <div class="confidence-text">Confidence: {(1-proba)*100:.1f}%</div>
            """, unsafe_allow_html=True)
            st.success("This URL appears to be legitimate")
        
        # Key indicators
        st.markdown("### Security Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">HTTPS Status</div>
                <div class="info-value">{'Not Secure' if features_dict.get('NoHttps', 0) == 1 else 'Secure'}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">URL Length</div>
                <div class="info-value">{features_dict.get('UrlLength', 0)} characters</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Suspicious Keywords</div>
                <div class="info-value">{features_dict.get('NumSensitiveWords', 0)} found</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Brand Impersonation</div>
                <div class="info-value">{'Detected' if features_dict.get('EmbeddedBrandName', 0) == 1 else 'None'}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Uses IP Address</div>
                <div class="info-value">{'Yes' if features_dict.get('IpAddress', 0) == 1 else 'No'}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Subdomain Level</div>
                <div class="info-value">{features_dict.get('SubdomainLevel', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed features
        with st.expander("View All Features"):
            df_show = pd.DataFrame([features_dict])
            st.dataframe(df_show.T, use_container_width=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem; border-top: 1px solid #333;">
    <p style="margin: 0; font-size: 0.9rem;">Deep Learning Phishing Detection System</p>
</div>
""", unsafe_allow_html=True)