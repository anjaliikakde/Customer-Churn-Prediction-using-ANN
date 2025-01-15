# Customer Churn Prediction Using Artificial Neural Networks  

This project leverages **Artificial Neural Networks (ANN)** to predict customer churn using the Churn_Modelling.csv dataset. Customer churn refers to the likelihood of customers leaving a service or discontinuing usage. By predicting churn, businesses can take proactive steps to retain customers and improve revenue. This is a classification problem with two possible outcomes:  
&star; The customer is likely to churn.  
&star; The customer is not likely to churn.  

---

## Dataset Information  
The dataset used in this project, Churn_Modelling.csv, contains 10,000 rows and 14 features. 
## Technology Used  
 #### Programming Language
 - Python
  #### Libraries 
   - TensorFlow  
   - Pandas  
   - NumPy  
   - Scikit-learn  
   - TensorBoard  
   - Matplotlib  
   - Streamlit  

---

## Model Architecture  
The ANN model is structured as follows:  
- Input Layer having 12 input features.  
- Hidden Layers: 
  -  H1 having 64 neurons, ReLU activation.  
  - H2 having 32 neurons, ReLU activation.  
- Output Layer having 1 neuron, Sigmoid activation.  
- Adam Optimizer is used for minimizing loss function. 
- Loss Function - Binary Crossentropy because it a classification problem.
- Metrics: MSA & MEA 

---

## Preprocessing Steps  
1. **Data Cleaning:**  
   Removed irrelevant columns such as RowNumber, CustomerId, and Surname.  
2. **Encoding Categorical Variables:**  
   Converted Geography and Gender into numerical format using one-hot encoding.  
3. **Feature Scaling:** 
   Standardized numerical features such as CreditScore and Balance.  
4. **Splitting the Dataset:** 
   Split the data into 80% training and 20% testing sets.  

---

## Getting Started

To run this project Write the commonds in terminal-

- Clone the repository using the command:  
```bash
   git clone <url> 
``` 
- Create a virtual environment using the command: 
  ``` bash
   conda create -p venv python==3.11 -y
  ```   
- Activate the virtual environment using the command: 
   ```bash
   conda activate venv/
   ```  
- Install dependencies using the command:  
  ``` bash
   pip install -r requirements.txt
   ```  
- Open the terminal and run `app.py` using the command:  
  ``` bash
   streamlit run app.py
   ```


## Read about Artificial Neural Networks 

 - [Artificial Neural Networks](https://medium.com/machine-learning-researcher/artificial-neural-network-ann-4481fa33d85a)
 

-[## Screenshots of Streamlit web app]()

![NewCapture](https://github.com/user-attachments/assets/1a1b4a68-766a-4ba6-8121-0e0390b9d976)
