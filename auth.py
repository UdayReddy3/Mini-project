"""
Authentication helper functions for Streamlit app
Manages session state and user authentication UI.
"""

import streamlit as st
from db import login_user, register_user, get_user_info
from language import get_translation


def t(key: str) -> str:
    """Translate a key using current session language; fallback to English."""
    lang = st.session_state.get('language', 'en')
    return get_translation(lang, key, key)


def init_session_state():
    """Initialize session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'auth_page' not in st.session_state:
        st.session_state.auth_page = 'register'  # 'register' or 'login'


def logout():
    """Logout the current user."""
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    # Use translated logout message if available
    logout_msg = t('btn_logout') if t('btn_logout') != 'btn_logout' else 'Logged out'
    st.success(f"✓ {logout_msg}!")


def show_login_page():
    """Display login and register page with professional styling."""
    # Add professional styling CSS
    st.markdown("""
    <style>
    .auth-main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem 0;
    }
    
    .auth-header {
        text-align: center;
        color: white;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .auth-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .auth-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .auth-card {
        background: white;
        border-radius: 15px;
        padding: 3rem 2.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 2rem auto;
        max-width: 500px;
    }
    
    .auth-subheader {
        font-size: 1.8rem;
        color: #2d3748;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .auth-description {
        color: #718096;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    
    .auth-divider {
        border: 1px solid #e2e8f0;
        margin: 2rem 0;
    }
    
    .auth-button-group {
        display: flex;
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .auth-footer {
        text-align: center;
        color: #718096;
        margin-top: 1.5rem;
        font-size: 0.95rem;
    }
    
    .auth-footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
    }
    
    .auth-features {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-top: 2rem;
        padding: 2rem;
        background: #f7fafc;
        border-radius: 12px;
    }
    
    .auth-feature-item {
        text-align: center;
    }
    
    .auth-feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .auth-feature-text {
        font-size: 0.9rem;
        color: #2d3748;
        font-weight: 500;
    }
    
    .demo-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        box-shadow: 0 10px 20px rgba(245, 87, 108, 0.2);
    }
    
    .demo-box h4 {
        margin-bottom: 1rem;
        font-size: 1rem;
    }
    
    .demo-box code {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.75rem;
        border-radius: 6px;
        display: block;
        margin-top: 0.5rem;
        font-weight: 600;
        border-left: 3px solid white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="auth-header">
        <h1>🌾 {t('app_title')}</h1>
        <p>{t('app_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show different pages based on auth_page state
    if st.session_state.auth_page == 'register':
        show_register_page()
    else:
        show_login_form()
    
    st.markdown(f"""
    <div style='margin: 3rem auto; max-width: 500px;'>
        <div class="auth-features">
            <div class="auth-feature-item">
                <div class="auth-feature-icon">🖼️</div>
                <div class="auth-feature-text">{t('upload_image')}</div>
            </div>
            <div class="auth-feature-item">
                <div class="auth-feature-icon">🤖</div>
                <div class="auth-feature-text">{t('predicted_disease')}</div>
            </div>
            <div class="auth-feature-item">
                <div class="auth-feature-icon">📊</div>
                <div class="auth-feature-text">{t('confidence_score')}</div>
            </div>
            <div class="auth-feature-item">
                <div class="auth-feature-icon">🔒</div>
                <div class="auth-feature-text">{t('btn_logout')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_register_page():
    """Display registration page with professional styling."""
    st.markdown("""
    <style>
    .register-card {
        background: white;
        border-radius: 15px;
        padding: 3rem 2.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 2rem auto;
        max-width: 500px;
    }
    
    .form-label {
        color: #2d3748;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        display: block;
    }
    
    .form-input {
        width: 100%;
        padding: 0.75rem;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        font-size: 1rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .form-input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    col_spacer1, col_card, col_spacer2 = st.columns([0.5, 1, 0.5])
    
    with col_card:
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='font-size: 1.8rem; font-weight: 700; color: #2d3748;'>📝 {t('create_account')}</p>
            <p style='color: #718096; margin-top: 0.5rem;'>{t('join_us')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        reg_username = st.text_input(t('username'), key="reg_username", placeholder=t('username'), help=t('username_help'))
        reg_email = st.text_input(t('email'), key="reg_email", placeholder=t('email_help'), help=t('email_help'))
        reg_full_name = st.text_input(t('full_name'), key="reg_full_name", placeholder=t('full_name_help'), help=t('full_name_help'))
        reg_password = st.text_input(t('password'), type="password", key="reg_password", placeholder=t('password_help'), help=t('password_help'))
        reg_confirm = st.text_input(t('confirm_password'), type="password", key="reg_confirm", placeholder=t('confirm_password'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            register_clicked = st.button(f"✅ {t('btn_register')}", use_container_width=True, key="register_btn", help=t('btn_register'))
        
        with col2:
            login_clicked = st.button(f"🔓 {t('btn_login')}", use_container_width=True, key="go_to_login", help=t('btn_go_login'))
        
        if register_clicked:
            if not reg_username or not reg_email or not reg_password:
                st.error(f"❌ {t('fill_required')}")
            elif len(reg_password) < 6:
                st.error(f"❌ {t('min_password')}")
            elif reg_password != reg_confirm:
                st.error(f"❌ {t('password_mismatch')}")
            else:
                success, message = register_user(reg_username, reg_email, reg_password, reg_full_name)
                
                if success:
                    st.success(f"✅ {t('registration_success')}")
                    st.success(t('proceed_login'))
                    st.session_state.auth_page = 'login'
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        if login_clicked:
            st.session_state.auth_page = 'login'
            st.rerun()
        
        st.markdown("---")
        st.info(f"✓ {t('have_account')} {t('btn_go_login')}")



def show_login_form():
    """Display login page with professional styling."""
    st.markdown("""
    <style>
    .login-card {
        background: white;
        border-radius: 15px;
        padding: 3rem 2.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 2rem auto;
        max-width: 500px;
    }
    
    .demo-credentials-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(245, 87, 108, 0.2);
    }
    
    .demo-credentials-box h4 {
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .demo-credentials-box code {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem;
        border-radius: 4px;
        display: inline-block;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col_spacer1, col_card, col_spacer2 = st.columns([0.5, 1, 0.5])
    
    with col_card:
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='font-size: 1.8rem; font-weight: 700; color: #2d3748;'>🔓 {t('welcome_back')}</p>
            <p style='color: #718096; margin-top: 0.5rem;'>{t('sign_in')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input(t('username'), key="login_username", placeholder=t('username_help'), help=t('username_help'))
        password = st.text_input(t('password'), type="password", key="login_password", placeholder=t('password_help'), help=t('password_help'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            login_clicked = st.button(f"✅ {t('btn_login')}", use_container_width=True, key="login_btn", help=t('btn_login'))
        
        with col2:
            register_clicked = st.button(f"📝 {t('btn_register')}", use_container_width=True, key="go_to_register", help=t('btn_go_register'))
        
        if login_clicked:
            if not username or not password:
                st.error(f"❌ {t('login_failed')}")
            else:
                success, message, user_id = login_user(username, password)
                
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.success(f"✅ {t('login_success')}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        if register_clicked:
            st.session_state.auth_page = 'register'
            st.rerun()
        
        st.markdown("---")
        
        st.markdown(f"""
        <div class="demo-credentials-box">
            <h4>🔑 {t('demo_credentials')}</h4>
            <code>demo</code> / <code>demo123</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"✓ {t('no_account')} {t('btn_go_register')}")


def show_user_profile():
    """Display user profile in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"### 👤 {st.session_state.username}")
        
        user_info = get_user_info(st.session_state.user_id)
        if user_info:
            st.markdown(f"**{t('email')}:** {user_info['email']}")
            if user_info['full_name']:
                st.markdown(f"**{t('full_name')}:** {user_info['full_name']}")
            st.markdown(f"**{t('step3')}:** {user_info['created_at'][:10]}")
        
        st.markdown("---")
        
        if st.button(f"🚪 {t('btn_logout')}", use_container_width=True):
            logout()
            st.rerun()
