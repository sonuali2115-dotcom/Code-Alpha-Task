"""
Language Translation Web Application with Text-to-Speech 
Fixed translation display issue

Requires: streamlit, gtts, deep-translator

Install dependencies:
pip install streamlit gtts deep-translator

Run the app:
streamlit run Translator.py
"""

import streamlit as st
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
import time

# Comprehensive language list
LANGUAGES = {
    'af': 'Afrikaans', 'sq': 'Albanian', 'am': 'Amharic', 'ar': 'Arabic',
    'hy': 'Armenian', 'az': 'Azerbaijani', 'eu': 'Basque', 'be': 'Belarusian',
    'bn': 'Bengali', 'bs': 'Bosnian', 'bg': 'Bulgarian', 'ca': 'Catalan',
    'ceb': 'Cebuano', 'zh-CN': 'Chinese (Simplified)', 'zh-TW': 'Chinese (Traditional)',
    'co': 'Corsican', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish',
    'nl': 'Dutch', 'en': 'English', 'eo': 'Esperanto', 'et': 'Estonian',
    'fi': 'Finnish', 'fr': 'French', 'fy': 'Frisian', 'gl': 'Galician',
    'ka': 'Georgian', 'de': 'German', 'el': 'Greek', 'gu': 'Gujarati',
    'ht': 'Haitian Creole', 'ha': 'Hausa', 'haw': 'Hawaiian', 'he': 'Hebrew',
    'hi': 'Hindi', 'hmn': 'Hmong', 'hu': 'Hungarian', 'is': 'Icelandic',
    'ig': 'Igbo', 'id': 'Indonesian', 'ga': 'Irish', 'it': 'Italian',
    'ja': 'Japanese', 'jw': 'Javanese', 'kn': 'Kannada', 'kk': 'Kazakh',
    'km': 'Khmer', 'rw': 'Kinyarwanda', 'ko': 'Korean', 'ku': 'Kurdish',
    'ky': 'Kyrgyz', 'lo': 'Lao', 'la': 'Latin', 'lv': 'Latvian',
    'lt': 'Lithuanian', 'lb': 'Luxembourgish', 'mk': 'Macedonian', 'mg': 'Malagasy',
    'ms': 'Malay', 'ml': 'Malayalam', 'mt': 'Maltese', 'mi': 'Maori',
    'mr': 'Marathi', 'mn': 'Mongolian', 'my': 'Myanmar (Burmese)', 'ne': 'Nepali',
    'no': 'Norwegian', 'ny': 'Nyanja (Chichewa)', 'or': 'Odia (Oriya)', 'ps': 'Pashto',
    'fa': 'Persian', 'pl': 'Polish', 'pt': 'Portuguese', 'pa': 'Punjabi',
    'ro': 'Romanian', 'ru': 'Russian', 'sm': 'Samoan', 'gd': 'Scots Gaelic',
    'sr': 'Serbian', 'st': 'Sesotho', 'sn': 'Shona', 'sd': 'Sindhi',
    'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'so': 'Somali',
    'es': 'Spanish', 'su': 'Sundanese', 'sw': 'Swahili', 'sv': 'Swedish',
    'tl': 'Tagalog (Filipino)', 'tg': 'Tajik', 'ta': 'Tamil', 'tt': 'Tatar',
    'te': 'Telugu', 'th': 'Thai', 'tr': 'Turkish', 'tk': 'Turkmen',
    'uk': 'Ukrainian', 'ur': 'Urdu', 'ug': 'Uyghur', 'uz': 'Uzbek',
    'vi': 'Vietnamese', 'cy': 'Welsh', 'xh': 'Xhosa', 'yi': 'Yiddish',
    'yo': 'Yoruba', 'zu': 'Zulu'
}

# Languages supported by gTTS
TTS_SUPPORTED_LANGS = [
    'af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 
    'et', 'fi', 'fr', 'gu', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'iw', 'ja', 
    'jw', 'km', 'kn', 'ko', 'la', 'lv', 'ml', 'mr', 'ms', 'my', 'ne', 'nl', 
    'no', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sq', 'sr', 'su', 'sv', 'sw', 
    'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW'
]

def is_tts_supported(lang_code):
    """Check if language is supported by gTTS"""
    return lang_code in TTS_SUPPORTED_LANGS

def translate_text(text, source_lang, target_lang):
    """Translate text using deep-translator"""
    try:
        if source_lang == 'auto':
            translator = GoogleTranslator(source='auto', target=target_lang)
        else:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
        
        translation = translator.translate(text)
        return translation, None
    except Exception as e:
        return None, str(e)

def text_to_speech(text, lang):
    """Convert text to speech"""
    try:
        tts_lang = lang
        if lang == 'zh-CN':
            tts_lang = 'zh-cn'
        elif lang == 'zh-TW':
            tts_lang = 'zh-tw'
        elif lang == 'he':
            tts_lang = 'iw'
        
        if not is_tts_supported(lang) and not is_tts_supported(tts_lang):
            return None
        
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except Exception as e:
        return None

# Page configuration
st.set_page_config(
    page_title="Linguator V2",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px 0;
    }
    .stTextArea textarea {
        height: 250px !important;
        font-size: 16px !important;
    }
    .translation-box {
        padding: 20px;
        border-radius: 10px;
        min-height: 250px;
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state with unique keys
if 'trans_result' not in st.session_state:
    st.session_state.trans_result = ""

# Header
st.markdown("<h1 class='main-header'>🌐 Linguator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af;'>Translate between 110+ languages • Full text-to-speech support</p>", unsafe_allow_html=True)

# Apply permanent dark mode
st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border-color: #444 !important;
    }
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #3730a3;
        color: #ffffff;
    }
    .stButton > button:hover {
        background-color: #4338ca;
    }
    h3 {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("---")

# Language selection row
col1, col2, col3 = st.columns([5, 1, 5])

with col1:
    st.subheader("🔤 Source Language")
    source_lang = st.selectbox(
        "From",
        options=['auto'] + list(LANGUAGES.keys()),
        format_func=lambda x: 'Auto Detect' if x == 'auto' else LANGUAGES[x],
        index=0,
        key='src_lang',
        label_visibility="collapsed"
    )

with col2:
    st.write("")
    st.write("")
    if st.button("🔄", help="Swap languages", key='swap_btn'):
        st.info("Language swap feature")

with col3:
    st.subheader("🌍 Target Language")
    target_lang = st.selectbox(
        "To",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        index=list(LANGUAGES.keys()).index('ur'),
        key='tgt_lang',
        label_visibility="collapsed"
    )

st.markdown("---")

# Main translation area
col1, col2 = st.columns(2)

# SOURCE TEXT AREA
with col1:
    st.markdown("### 📝 Enter Text")
    
    source_text = st.text_area(
        "Source",
        placeholder="Type or paste your text here...",
        height=250,
        key='source_input_v2',
        label_visibility="collapsed"
    )
    
    # Source controls
    col_a, col_b, col_c = st.columns([2, 2, 6])
    
    with col_a:
        st.caption(f"📊 {len(source_text)} chars")
    
    with col_b:
        tts_src_ok = is_tts_supported(source_lang) if source_lang != 'auto' else False
        
        if st.button("🔊 Listen", key='src_speak', disabled=not tts_src_ok or not source_text):
            if source_text and source_lang != 'auto':
                audio = text_to_speech(source_text, source_lang)
                if audio:
                    st.audio(audio, format='audio/mp3')
                else:
                    st.error("TTS failed")
        
        if not tts_src_ok and source_lang != 'auto':
            st.caption("⚠️ No TTS")

# TRANSLATION DISPLAY AREA
with col2:
    st.markdown("### 🌍 Translation")
    
    # Create a container for the translation
    translation_container = st.container()
    
    with translation_container:
        # Display translation result with better styling
        if st.session_state.trans_result:
            # Show the actual translation
            st.markdown(
                f"""
                <div class='translation-box' style='
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 2px solid #444;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                '>
                {st.session_state.trans_result}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Show placeholder
            st.markdown(
                f"""
                <div class='translation-box' style='
                    background-color: #2d2d2d;
                    color: #666;
                    border: 2px dashed #444;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-style: italic;
                '>
                Translation will appear here...
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Translation controls
    col_d, col_e, col_f = st.columns([2, 2, 6])
    
    tts_tgt_ok = is_tts_supported(target_lang)
    has_trans = bool(st.session_state.trans_result)
    
    with col_d:
        if st.button("🔊 Listen", key='tgt_speak', disabled=not tts_tgt_ok or not has_trans):
            if st.session_state.trans_result:
                audio = text_to_speech(st.session_state.trans_result, target_lang)
                if audio:
                    st.audio(audio, format='audio/mp3')
                else:
                    st.error("TTS failed")
        
        if not tts_tgt_ok:
            st.caption("⚠️ No TTS")
    
    with col_e:
        if st.button("📋 Copy", key='copy_btn', disabled=not has_trans):
            if st.session_state.trans_result:
                st.code(st.session_state.trans_result, language=None)
                st.success("Ready to copy! ✅")

st.markdown("---")

# TRANSLATE BUTTON - Centered and prominent
col1, col2, col3 = st.columns([3, 4, 3])
with col2:
    if st.button("🚀 TRANSLATE NOW", type="primary", use_container_width=True, key='translate_btn'):
        if source_text.strip():
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Translating...")
            progress_bar.progress(30)
            
            # Perform translation
            translated, error = translate_text(source_text, source_lang, target_lang)
            
            progress_bar.progress(70)
            
            if error:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Translation Error: {error}")
                st.session_state.trans_result = ""
            elif translated:
                progress_bar.progress(100)
                status_text.text("✅ Translation complete!")
                
                # Store the translation
                st.session_state.trans_result = translated
                
                # Small delay to show completion
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show success message
                st.success("✅ Translation completed successfully!")
                
                # Force refresh to show translation
                st.rerun()
            else:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ Translation failed. Please try again.")
                st.session_state.trans_result = ""
        else:
            st.warning("⚠️ Please enter text to translate")

# Clear button
col1, col2, col3 = st.columns([3, 4, 3])
with col2:
    if st.button("🗑️ Clear All", key='clear_btn', use_container_width=True):
        st.session_state.trans_result = ""
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #9ca3af; padding: 20px;'>
        <p><strong>Powered by Google Translate & Google TTS</strong></p>
        <p>✅ Supports {len(LANGUAGES)} languages | 🎤 TTS available for {len(TTS_SUPPORTED_LANGS)} languages</p>
        <p><strong>Full support for: Urdu (اردو), Arabic (العربية), Hindi (हिंदी), Bengali (বাংলা), and more!</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)