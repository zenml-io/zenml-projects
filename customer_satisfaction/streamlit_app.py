from re import S
import streamlit as st
from inference.predict import predict

def main(): 
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Customer Satisfaction Demo App </h1>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html = True)

    payment_sequential = st.sidebar.slider("Payment Sequential") 
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.number_input("Payment vSalue") 
    price = st.number_input("Price")  
    freight_value = st.number_input("freight_value") 
    product_name_lenght = st.number_input("product_name_lenght")  
    product_description_lenght = st.number_input("product_description_lenght") 
    product_photos_qty = st.number_input("product_photos_qty") 
    product_weight_g = st.number_input("product_weight_g") 
    product_length_cm = st.number_input("product_length_cm") 
    product_height_cm = st.number_input("product_height_cm")  
    product_width_cm = st.number_input("product_width_cm")

    result = "" 
    if st.button("Predict"):
        result = predict(payment_sequential,
payment_installments,
payment_value,	
price,	
freight_value,	
product_name_lenght	,
product_description_lenght,	
product_photos_qty,	
product_weight_g,	
product_length_cm,
product_height_cm,
product_width_cm) 
    st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()


