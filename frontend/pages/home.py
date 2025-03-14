# import streamlit as st
 
# # Must be the first Streamlit command
# st.set_page_config(
#     page_title="DataNexus Pro | Home",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
 
# # Custom CSS for consistent styling
# st.markdown("""
#     <style>
#     /* Global styles */
#     .stApp {
#         background-color: white;
#     }
    
#     /* Sidebar styling */
#     [data-testid="stSidebar"] {
#         background-color: white;
#     }
    
#     /* Navigation items in sidebar */
#     .css-17lntkn {
#         color: #A9A9A9 !important;
#         font-weight: normal;
#     }
    
#     /* Selected navigation item */
#     .css-17lntkn.selected {
#         background-color: rgba(240, 242, 246, 0.5) !important;
#         border-radius: 4px;
#     }
    
#     /* Main content text */
#     h1, h2, h3, p {
#         color: black !important;
#     }
    
#     /* Feature cards */
#     .feature-card {
#         background-color: #f8f9fa;
#         border-radius: 8px;
#         padding: 20px;
#         margin: 10px 0;
#         border: 1px solid #eee;
#     }
    
#     .feature-card h3 {
#         color: black !important;
#         margin-bottom: 10px;
#     }
    
#     .feature-card p {
#         color: #666 !important;
#     }
    
#     /* Stats container */
#     .stats-container {
#         background-color: #f8f9fa;
#         border-radius: 8px;
#         padding: 20px;
#         margin: 20px 0;
#         border: 1px solid #eee;
#     }
    
#     /* Metric styling */
#     [data-testid="stMetricValue"] {
#         color: black !important;
#     }
    
#     /* Button styling */
#     .stButton > button {
#         background-color: white;
#         color: black;
#         border: 1px solid #ddd;
#         border-radius: 4px;
#         padding: 0.5rem 1.5rem;
#         width: 100%;
#     }
    
#     .stButton > button:hover {
#         background-color: #f8f9fa;
#         border-color: #ddd;
#     }
 
#     /* Make header white */
#     header[data-testid="stHeader"] {
#         background-color: white;
#     }
    
#     /* Remove any dark backgrounds */
#     .stApp header {
#         background-color: white !important;
#     }
    
#     /* Style header elements */
#     .stApp header button {
#         color: black !important;
#     }
    
#     /* Sidebar navigation styling */
#     [data-testid="stSidebar"] {
#         background-color: white;
#     }
    
#     /* Remove "data extraction" from top navigation */
#     .css-17lntkn[aria-label="data extraction"] {
#         display: none !important;
#     }
    
#     /* Navigation items in sidebar */
#     .css-17lntkn {
#         color: #A9A9A9 !important;
#         font-weight: normal;
#     }
    
#     /* Quick Navigation buttons */
#     .stButton > button {
#         background-color: transparent;
#         color: black;
#         border: none;
#         text-align: left;
#         padding: 8px 0;
#         font-weight: normal;
#     }
    
#     .stButton > button:hover {
#         background-color: rgba(240, 242, 246, 0.5);
#         border-radius: 4px;
#     }
#     </style>
# """, unsafe_allow_html=True)
 
# # Initialize session state if needed
# if 'welcome_shown' not in st.session_state:
#     st.session_state.welcome_shown = False
 
# # Main content
# st.markdown("""
#     <div style='padding: 1rem 0;'>
#         <h1 style='color: black; font-size: 2.5rem; font-weight: 500;'>📊 DataNexus Pro</h1>
#         <p style='color: #666; font-size: 1.2rem;'>Unified Data Extraction Platform</p>
#     </div>
# """, unsafe_allow_html=True)
 
# # Platform overview section
# st.markdown("### Transform Your Data Processing")
# st.markdown("""
#     DataNexus Pro simplifies the extraction and analysis of data from various sources.
#     Our platform provides advanced tools for both PDF extraction and web scraping,
#     making data processing more efficient than ever.
# """)
 
# # Feature highlights
# col1, col2, col3 = st.columns(3)
 
# with col1:
#     st.markdown("""
#         <div class='feature-card'>
#             <h3>📄 PDF Processing</h3>
#             <p>Extract text, tables, and images from PDF documents with high accuracy</p>
#             <ul style='color: #666;'>
#                 <li>Table extraction</li>
#                 <li>Image extraction</li>
#                 <li>Text analysis</li>
#             </ul>
#         </div>
#     """, unsafe_allow_html=True)
 
# with col2:
#     st.markdown("""
#         <div class='feature-card'>
#             <h3>🌐 Web Scraping</h3>
#             <p>Collect data from websites efficiently and reliably</p>
#             <ul style='color: #666;'>
#                 <li>Structured data extraction</li>
#                 <li>Dynamic content handling</li>
#                 <li>Custom scraping rules</li>
#             </ul>
#         </div>
#     """, unsafe_allow_html=True)
 
# with col3:
#     st.markdown("""
#         <div class='feature-card'>
#             <h3>📊 Analytics</h3>
#             <p>Transform raw data into actionable insights</p>
#             <ul style='color: #666;'>
#                 <li>Data visualization</li>
#                 <li>Statistical analysis</li>
#                 <li>Export capabilities</li>
#             </ul>
#         </div>
#     """, unsafe_allow_html=True)
 
# # Platform statistics
# st.markdown("### Platform Statistics")
# st.markdown("""
#     <div class='stats-container'>
#         <div style='display: flex; justify-content: space-between;'>
# """, unsafe_allow_html=True)
 
# col1, col2, col3, col4 = st.columns(4)
 
# with col1:
#     st.metric("Documents Processed", "10K+")
# with col2:
#     st.metric("Active Users", "500+")
# with col3:
#     st.metric("Time Saved", "1000+ hrs")
# with col4:
#     st.metric("Accuracy Rate", "99.9%")
 
# # Getting Started section
# st.markdown("### Getting Started")
# st.markdown("""
#     <div class='feature-card'>
#         <p>Follow these simple steps to begin:</p>
#         <ol style='color: #666;'>
#             <li>Navigate to <b>Data Extraction</b> in the sidebar</li>
#             <li>Choose your extraction type (PDF or Web)</li>
#             <li>Select processing engine based on your needs</li>
#             <li>Upload your document or enter URL</li>
#             <li>View and analyze your results</li>
#         </ol>
#     </div>
# """, unsafe_allow_html=True)
 
# # Sidebar
# with st.sidebar:
    
#     st.markdown("### Quick Navigation")
 
#     st.markdown("""
#         <style>
#         /* Style the navigation buttons to look like links */
#         .nav-button {
#             background: none;
#             border: none;
#             color: #666;
#             padding: 8px 0;
#             width: 100%;
#             text-align: left;
#             cursor: pointer;
#         }
#         .nav-button:hover {
#             background-color: rgba(240, 242, 246, 0.5);
#             border-radius: 4px;
#         }
#         </style>
#     """, unsafe_allow_html=True)
    
#     if st.button("📥 Start Extraction"):
#         st.switch_page("pages/data_extraction")
    
   
 
import streamlit as st
 
# Must be the first Streamlit command
st.set_page_config(
    page_title="DataNexus Pro | Home",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# Custom CSS for consistent styling
st.markdown("""
    <style>
    /* Global styles */
    .stApp {
        background-color: white;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: white;
    }
    
    /* Navigation items in sidebar */
    .css-17lntkn {
        color: #A9A9A9 !important;
        font-weight: normal;
    }
    
    /* Selected navigation item */
    .css-17lntkn.selected {
        background-color: rgba(240, 242, 246, 0.5) !important;
        border-radius: 4px;
    }
    
    /* Main content text */
    h1, h2, h3, p {
        color: black !important;
    }
    
    /* Feature cards */
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #eee;
    }
    
    .feature-card h3 {
        color: black !important;
        margin-bottom: 10px;
    }
    
    .feature-card p {
        color: #666 !important;
    }
    
    /* Stats container */
    .stats-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid #eee;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: black !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: white;
        color: black;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #f8f9fa;
        border-color: #ddd;
    }
 
    /* Make header white */
    header[data-testid="stHeader"] {
        background-color: white;
    }
    
    /* Remove any dark backgrounds */
    .stApp header {
        background-color: white !important;
    }
    
    /* Style header elements */
    .stApp header button {
        color: black !important;
    }
    
    /* Sidebar navigation styling */
    [data-testid="stSidebar"] {
        background-color: white;
    }
    
    /* Remove "data extraction" from top navigation */
    .css-17lntkn[aria-label="data extraction"] {
        display: none !important;
    }
    
    /* Navigation items in sidebar */
    .css-17lntkn {
        color: #A9A9A9 !important;
        font-weight: normal;
    }
    
    /* Quick Navigation buttons */
    .stButton > button {
        background-color: transparent;
        color: black;
        border: none;
        text-align: left;
        padding: 8px 0;
        font-weight: normal;
    }
 
 
 
    /* Fix for the uploaded file name color in the file uploader */
[data-testid="stFileUploader"] div {
    color: black !important; /* Ensure the file name is visible */
    font-weight: 500;        /* Adjust font weight as needed */
}
 
/* Adjust the input and dropdown text color */
.stTextInput, .stSelectbox {
    color: black !important;
    background-color: white !important;
}
 
 
 
 
 
 
 
/* Ensure that all text within the sidebar is visible */
[data-testid="stSidebar"] * {
    color: black !important;
}
 
/* General fix for button and interactive element text */
.stButton > button, .stRadio > div, .stSelectbox > div {
    color: black !important;
    background-color: white !important;
}
 
/* Specific styling for the file uploader */
.stFileUploader {
    background-color: #f8f9fa;  /* Light background for better visibility */
    border: 1px solid #ddd;      /* Light border */
    border-radius: 4px;
}
 
    
 
 
 
 
    .stButton > button:hover {
        background-color: rgba(240, 242, 246, 0.5);
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)
 
# Initialize session state if needed
if 'welcome_shown' not in st.session_state:
    st.session_state.welcome_shown = False
 
# Main content
st.markdown("""
    <div style='padding: 1rem 0;'>
        <h1 style='color: black; font-size: 2.5rem; font-weight: 800;'>📊 DataNexus Pro</h1>
        <p style='color: #666; font-size: 1.2rem;'>Unified Data Extraction Platform</p>
    </div>
""", unsafe_allow_html=True)
 
# Platform overview section
st.markdown("### Transform Your Data Processing")
st.markdown("""
    DataNexus Pro simplifies the extraction and analysis of data from various sources.
    Our platform provides advanced tools for both PDF extraction and web scraping,
    making data processing more efficient than ever.
""")
 
# Feature highlights
col1, col2, col3 = st.columns(3)
 
with col1:
    st.markdown("""
        <div class='feature-card'>
            <h3>📄 PDF Processing</h3>
            <p>Extract text, tables, and images from PDF documents with high accuracy</p>
            <ul style='color: #666;'>
                <li>Table extraction</li>
                <li>Image extraction</li>
                <li>Text analysis</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
 
with col2:
    st.markdown("""
        <div class='feature-card'>
            <h3>🌐 Web Scraping</h3>
            <p>Collect data from websites efficiently and reliably</p>
            <ul style='color: #666;'>
                <li>Structured data extraction</li>
                <li>Dynamic content handling</li>
                <li>Custom scraping rules</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
 
with col3:
    st.markdown("""
        <div class='feature-card'>
            <h3>📊 Analytics</h3>
            <p>Transform raw data into actionable insights</p>
            <ul style='color: #666;'>
                <li>Data visualization</li>
                <li>Statistical analysis</li>
                <li>Export capabilities</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
 
# Platform statistics
st.markdown("### Platform Statistics")
st.markdown("""
    <div class='stats-container'>
        <div style='display: flex; justify-content: space-between;'>
""", unsafe_allow_html=True)
 
col1, col2, col3, col4 = st.columns(4)
 
with col1:
    st.metric("Documents Processed", "10K+")
with col2:
    st.metric("Active Users", "500+")
with col3:
    st.metric("Time Saved", "1000+ hrs")
with col4:
    st.metric("Accuracy Rate", "99.9%")
 
# Getting Started section
st.markdown("### Getting Started")
st.markdown("""
    <div class='feature-card'>
        <p>Follow these simple steps to begin:</p>
        <ol style='color: #666;'>
            <li>Navigate to <b>Data Extraction</b> in the sidebar</li>
            <li>Choose your extraction type (PDF or Web)</li>
            <li>Select processing engine based on your needs</li>
            <li>Upload your document or enter URL</li>
            <li>View and analyze your results</li>
        </ol>
    </div>
""", unsafe_allow_html=True)
 
# Sidebar
with st.sidebar:
    
    st.markdown("### Quick Navigation")
 
    st.markdown("""
        <style>
        /* Style the navigation buttons to look like links */
        .nav-button {
            background: none;
            border: none;
            color: #666;
            padding: 8px 0;
            width: 100%;
            text-align: left;
            cursor: pointer;
        }
        .nav-button:hover {
            background-color: rgba(240, 242, 246, 0.5);
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if st.button("📥 Start Extraction"):
        st.switch_page("working_app.py")
 