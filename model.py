import streamlit as st
import pandas as pd
import pickle
import base64


df = pd.read_csv(r"water_pollution_disease.csv")
# Page settings
st.set_page_config(page_title="Water & Disease Risk Predictor", layout="centered")

st.markdown("<h1 style='text-align: center;'>🌊 Water & Disease Risk Predictor</h1>", unsafe_allow_html=True)


def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background(r"pro_3.jpg")




# Load models
with open('diarrheal_cases_model.pkl', 'rb') as f: diarrheal_model = pickle.load(f)
with open('cholera_cases_model.pkl', 'rb') as f: cholera_model = pickle.load(f)
with open('infant_mortality_model.pkl', 'rb') as f: infant_model = pickle.load(f)
with open('disease_risk_model1.pkl', 'rb') as f: disease_model = pickle.load(f)
with open('water_safe_model1.pkl', 'rb') as f: water_model = pickle.load(f)




st.sidebar.title("🌊 Input Features")


df = pd.read_csv(r"water_pollution_disease.csv")
df.columns = df.columns.str.strip().str.lower()  # normalize column names


country_list = sorted(df['country'].dropna().unique())
region_list = sorted(df['region'].dropna().unique())

# Collect input values from sidebar
country = st.sidebar.selectbox("Country", country_list)
region = st.sidebar.selectbox("Region", region_list)
water_source_type = st.sidebar.selectbox("Water Source Type", ['River', 'Well', 'Tap'])
water_treatment_method = st.sidebar.selectbox("Treatment Method", ['Boiling', 'Filtration', 'None'])
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 100.0, 10.0)
dissolved_oxygen = st.sidebar.slider("Dissolved Oxygen (mg/L)", 0.0, 14.0, 7.0)
nitrate_level = st.sidebar.slider("Nitrate Level (mg/L)", 0.0, 50.0, 10.0)
contaminant_level = st.sidebar.slider("Contaminant Level (ppm)", 0.0, 500.0, 100.0)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
rainfall = st.sidebar.slider("Rainfall (mm/year)", 0.0, 3000.0, 1000.0)
bacteria_count = st.sidebar.slider("Bacteria Count (cfu/ml)", 0, 10000, 500)
lead_concentration = st.sidebar.slider("Lead Concentration (µg/l)", 0.0, 100.0, 10.0)
gdp = st.sidebar.number_input("GDP per Capita", 0, 100000, 5000)
healthcare = st.sidebar.slider("Healthcare Access Index", 0, 100, 70)
clean_water_access = st.sidebar.slider("Access to Clean Water (%)", 0, 100, 80)
sanitation = st.sidebar.slider("Sanitation Coverage (%)", 0, 100, 75)
urbanization = st.sidebar.slider("Urbanization Rate (%)", 0, 100, 60)
density = st.sidebar.slider("Population Density (people/km²)", 0, 5000, 300)

# Set dummy targets for intermediate models
diarrheal_cases = 0
cholera_cases = 0
typhoid_cases = 0
infant_mortality = 0
water_safety = "Safe"
disease_risk_level = "High"

# Feature selection
option = st.selectbox("Select Prediction Type", ['Disease Risk Level', 'Cholera Cases', 'Infant Mortality Rate', 'Diarrheal Cases', 'Water Safety'])

# Fill full feature set
features = pd.DataFrame([{
    'country': country,
    'region': region,
    'water_source_type': water_source_type,
    'water_treatment_method': water_treatment_method,
    'ph_level': ph,
    'turbidity_(ntu)': turbidity,
    'dissolved_oxygen_(mg/l)': dissolved_oxygen,
    'nitrate_level_(mg/l)': nitrate_level,
    'contaminant_level_(ppm)': contaminant_level,
    'temperature_(°c)': temperature,
    'rainfall_(mm_per_year)': rainfall,
    'bacteria_count_(cfu/ml)': bacteria_count,
    'lead_concentration_(µg/l)': lead_concentration,
    'gdp_per_capita_(usd)': gdp,
    'healthcare_access_index_(0-100)': healthcare,
    'access_to_clean_water_(%_of_population)': clean_water_access,
    'sanitation_coverage_(%_of_population)': sanitation,
    'urbanization_rate_(%)': urbanization,
    'population_density_(people_per_km²)': density,
    'diarrheal_cases_per_100,000_people': diarrheal_cases,
    'cholera_cases_per_100,000_people': cholera_cases,
    'typhoid_cases_per_100,000_people': typhoid_cases,
    'infant_mortality_rate_(per_1,000_live_births)': infant_mortality,
    'water_safety': water_safety,
    'disease_risk_level': disease_risk_level
}])

# Apply prediction
if st.button("🔍 Predict"):
    if option == 'Disease Risk Level':
        pred = disease_model.predict(features)[0]
    elif option == 'Cholera Cases':
        pred = cholera_model.predict(features)[0]
    elif option == 'Infant Mortality Rate':
        pred = infant_model.predict(features)[0]
    elif option == 'Diarrheal Cases':
        pred = diarrheal_model.predict(features)[0]
    elif option == 'Water Safety':
        pred = water_model.predict(features)[0]

    st.markdown(f"<h2 style='text-align:center;'>🔷 Prediction: {pred}</h2>", unsafe_allow_html=True)

     # Show Recommendations
    # Recommendations based on prediction output
    st.markdown(
    """
    <style>
    /* Expander Header */
    .streamlit-expander > summary {
        background-color: white !important;
        color: black !important;
        font-weight: bold !important;
        padding: 10px !important;
        border-radius: 10px !important;
        font-size: 800px !important;
    }
    /* Expander Content */
    .streamlit-expander div[role="group"] {
        background-color: white !important;
        color: black !important;
        padding: 10px !important;
        font-size: 100px !important;
        font-weight: bold !important;
    }
    /* Expander open arrow customization */
    details[open] > summary::after {
        filter: brightness(0%);
    }
    /* Expander box styling */
    details {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


    if option == 'Disease Risk Level':

        # Assuming 'predicted_risk' is your model's prediction output
# predicted_risk = pipeline.predict(user_input_df)[0]

        

        if pred == "Low":
                st.markdown("### ✅ **Disease Risk Level: LOW**")
                st.info("The current water quality and health indicators suggest a **low risk** of waterborne diseases. Keep up the good practices!")

                with st.expander("**🧼 Recommended Ongoing Practices**"):
                    st.markdown("""
                    - ##### Continue using clean and treated water.  
                    - ##### Maintain good personal and community hygiene.  
                    - ##### Regularly clean water tanks and storage units.  
                    - ##### Educate others about safe water usage.
                    """)

                with st.expander("📚 **WHY PREVCENTION STILL MATTERS**"):
                    st.warning("""
                    ##### Even in low-risk areas, infrastructure failures or natural events (like floods) can increase disease risk. Prevention is always better than cure.
                    """)

        elif pred == "Medium":
                st.markdown("### ⚠️ **Disease Risk Level: MODERATE**")
                st.warning("There is a **moderate risk** of waterborne diseases. It’s important to take preventive action now.")

                # First set of columns (Health + Disease info)
                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("🛡️ **HEALTH RECOMMENDATIONS**"):
                        st.info("""
                        - ##### Boil or purify water before drinking.  
                        - ##### Avoid open defecation and maintain sanitation.  
                        - ##### Use clean containers for water storage.   
                        - ##### Seek medical help at first sign of fever, diarrhea, or vomiting.  
                        """)

                with col2:
                    with st.expander("🦠 **COMMON WATERBRONE DISEASES**"):
                        st.markdown("""
                        - **Cholera**: Severe diarrhea and dehydration.  
                        - **Typhoid**:  High fever, weakness, and stomach pain.  
                        - **Hepatitis A**:  Affects the liver, causes jaundice.  
                        - **Amoebiasis**:  Cramps and prolonged diarrhea.  
                        - **Giardiasis**:  Fatigue, gas, and watery diarrhea.  
                        - **Cryptosporidiosis**:  Nausea and watery diarrhea.
                        """)

                # Divider
                st.divider()

                # Second set of columns (Testing Tips + Community Action)
                col3, col4 = st.columns(2)

                with col3:
                    with st.expander("🧪 **WATER QUALITY TESTING TIPS**"):
                        st.info("""
                        - ##### Use test strips or water testing kits to check pH, turbidity, and chlorine levels.  
                        - ##### Smell and taste the water – foul odor or unusual taste may indicate contamination.  
                        - ##### Report any visible pollutants or suspicious water color to local health officials.  
                        - ##### Encourage routine water sampling in schools, homes, and public places.
                        """)

                with col4:
                    with st.expander("🏘️ **COMMUNITY ACTION STEPS**"):
                        st.warning("""
                        - ##### Organize awareness drives about safe water practices.  
                        - ##### Collaborate with local NGOs for water purification initiatives.  
                        - ##### Promote rainwater harvesting and groundwater recharge.  
                        - ##### Ensure regular maintenance of local water infrastructure.
                        """)


        elif pred == "High":
                st.markdown("### 🚨 **Disease Risk Level: HIGH**")
                st.error("There is a **high risk** of waterborne diseases in this area. Immediate action is strongly recommended.")

                with st.expander("🚑 Immediate Steps to Follow"):
                    st.warning("""
                    - Do NOT drink untreated or suspicious water.  
                    - Disinfect water using boiling, chlorine, or filtration.  
                    - Use ORS (Oral Rehydration Salts) if anyone shows signs of dehydration.  
                    - Inform local health authorities to investigate water quality.  
                    - Promote emergency sanitation measures in the area.
                    """)

                with st.expander("📢 Urgent Help & Awareness"):
                    st.markdown("""
                    - ##### Contact local health officials immediately.  
                    - ##### Launch awareness drives in your community.  
                    - ##### Use SMS or community radios to spread urgent alerts.  
                    - ##### Set up temporary water purification and aid stations if needed.
                    """)


    elif option == 'Cholera Cases':
        with st.expander("**CAUSES** "):
            st.write("- ##### Cholera is an acute diarrheal infection caused by the bacterium Vibrio cholerae")
            st.write("- ##### Contaminated food and drink")
            st.write("- ##### Reduced level of stomach acid")

        with st.expander("🤒 **SYMPTOMS**"):
            st.write("- ##### Diarrhea")
            st.write("- ##### Nausea and Vomiting")
            st.write("- ##### Restlessness or irritability")
            st.write("- ##### Dehydration (ranging from mild to severe)")

        with st.expander("⚠️**PRECAUTIONS**"):
            st.write("- ##### Do not eat raw or half cooked meat")
            st.write("- ##### Drink treated purified water")
            st.write("- ##### Avoid street food")

        with st.expander("💊 **MEDICATIONS / TREATMENTS**"):
            st.write("- ##### drinking plenty of fluids or getting intravenous fluids to prevent dehydration")
            st.write("-  ##### antibiotics like doxycycline, erythromycin or azithromycin to help you feel better.")

    elif option == 'Diarrheal Cases':
            st.subheader("**CAUSES**")
            causes = """
            <div style='background-color:#f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #d3d3d3; font-size:16px;'>
                <ul style='margin: 0; padding-left: 20px;'>
                    <li><b>Bacteria like E. coli, Salmonella, and Vibrio cholerae</b></li>
                    <li><b>These bacteria are often found in untreated sewage and can easily contaminate drinking water</b></li>
                </ul>
            </div>
            """
            st.markdown(causes, unsafe_allow_html=True)

            st.subheader("🤒 **SYMPTOMS**")
            symptoms = """
            <div style='background-color:#f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #d3d3d3; font-size:16px;'>
                <ul style='margin: 0; padding-left: 20px;'>
                    <li><b>Fever</b></li>
                    <li><b>Abdominal pain</b></li>
                    <li><b>Gastrointestinal infections</b></li>
                </ul>
            </div>
            """
            st.markdown(symptoms, unsafe_allow_html=True)

            st.subheader("⚠️ **PRECAUTIONS**")
            precautions = """
            <div style='background-color:#fff3cd; padding: 15px; border-radius: 10px; border: 1px solid #ffeeba; font-size:16px;'>
                <ul style='margin: 0; padding-left: 20px;'>
                    <li><b>Foods to avoid:</b> Spicy foods, Milk and dairy, Citrus fruits, Coffee</li>
                    <li><b>Foods that help stop diarrhea:</b> Bananas, Rice, Applesauce</li>
                </ul>
            </div>
            """
            st.markdown(precautions, unsafe_allow_html=True)

            st.subheader("💊 **Medications / Treatments**")
            medications = """
            <div style='background-color:#e2f0d9; padding: 15px; border-radius: 10px; border: 1px solid #c3e6cb; font-size:16px;'>
                <ul style='margin: 0; padding-left: 20px;'>
                    <li><b>loperamide (Imodium)</b></li>
                    <li><b>bismuth subsalicylate (Pepto-Bismol and Kaopectate)</b></li>
                </ul>
            </div>
            """
            st.markdown(medications, unsafe_allow_html=True)


            

    elif option == 'Water Safety':
    
            if pred == 'Unsafe':
                st.warning("### 🏛️ Government Action Recommendations:")
                st.warning("""
                - ###### Implement or upgrade water treatment plants in the affected areas.
                - ###### Increase monitoring of water sources for contaminants like bacteria, lead, and nitrates.
                - ###### Launch awareness campaigns on waterborne diseases and sanitation practices.
                - ###### Invest in infrastructure to improve access to clean water and sanitation facilities.
                """)

            st.success("### 🙋‍♀️ What Individuals Can Do:")
            st.success("""
            - ###### Boil or filter water before drinking or cooking. 
            - ###### Use certified water purifiers and avoid drinking from unsafe sources.
            - ###### Report any signs of contamination to local authorities.
            - ###### Maintain personal hygiene and safe food practices to prevent infections.
            """)

            st.info("### 📞 Report Contaminated Water")
            st.info("""
            If you suspect water contamination in your area, here are the steps you can take:
            - **Call Jal Jeevan Mission Helpline**: `14420` (for rural drinking water issues)
            - **Visit Your Local Municipal Office** or Panchayat to lodge a complaint with the Health or Water Department.
            - **Email the Central Pollution Control Board (CPCB)**: `info.cpcb@nic.in`
            - **Use the Swachhata App** or **mSeva Water Quality App** to report issues through mobile.
            - **Contact State Pollution Control Boards**: Each state has its own portal for complaints (e.g., Telangana: [tspcb.cgg.gov.in](https://tspcb.cgg.gov.in)).
            - **File an RTI** (Right to Information) request if action is delayed or denied.
            💡 *Include location details, photos/videos, and any evidence to strengthen your report.*
            """)






    elif option == 'Infant Mortality Rate':
        st.header("👶 Infant Mortality & Water Pollution Impact")
        st.info("""
    💡 **Did You Know?**
    -  Unsafe water is responsible for **over 800,000 child deaths annually**.
    -  **Breastfed infants** have a significantly lower risk of infection from contaminated water.
    -   The **World Health Organization (WHO)** recommends universal access to safe drinking water to reduce infant mortality worldwide.
    """)

        with st.expander("💥 **CAUSES**"):
            st.write("""
            - **Waterborne Diseases:** Caused by bacteria, viruses, and protozoa in polluted water. 
            Infants are especially vulnerable to dehydration and malnutrition from repeated diarrhea episodes.
            - **Contaminated Food & Drink:** Unsafe water used in baby formula or food preparation increases infection risk.
            - **Heavy Metal Exposure:** Lead, arsenic, and nitrates in water can lead to developmental delays and organ damage.
            - **Reduced Immunity:** Infants have lower immunity and lower stomach acid levels, making it easier for pathogens to infect.
            """)

        with st.expander("🤒 **SYMPTOMS**"):
            st.write("""
            - **Diarrhea**
            - **Nausea and Vomiting**
            - **Restlessness or Irritability**
            - **Dehydration** (ranging from mild to severe)
            - **Fever or Signs of Infection**
            """)

        with st.expander("⚠️ **PRECAUTIONS**"):
            st.write("""
            - **Always boil or treat water** before using it for infants.
            - **Avoid raw or undercooked meat and eggs.**
            - **Use water filters** (like RO or UV systems) for safe drinking water.
            - **Avoid street food and unwashed produce.**
            - **Promote exclusive breastfeeding** in the first six months.
            """)

        with st.expander("💊 **MEDICATIONS / TREATMENTS**"):
            st.write("""
            - **Oral Rehydration Therapy (ORT)** or intravenous fluids to prevent or treat dehydration.
            - **Zinc Supplements** to reduce duration and severity of diarrhea.
            - **Antibiotics** such as doxycycline, erythromycin, or azithromycin for bacterial infections (only under medical supervision).
            - **Medical attention** is crucial if symptoms persist more than 24 hours or worsen.
            """)
