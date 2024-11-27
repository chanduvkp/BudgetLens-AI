BudgetLens-AI: Smart Budget Analytics Dashboard
Overview
BudgetLens-AI is an intelligent budget analysis dashboard that combines traditional data visualization with artificial intelligence to provide deep insights into organizational budgets. This interactive web application helps stakeholders understand budget distributions, trends, and patterns while leveraging LLM technology for automated analysis.
Key Features

Interactive Visualizations

Trend Analysis: Track budget changes over time with interactive line charts
Distribution Analysis: Understand budget allocation across departments using pie charts
Heat Map View: Visualize budget intensity patterns across departments and fiscal years
Department Comparison: Compare departmental budgets with horizontal bar charts


AI-Powered Insights

Automated budget analysis using Llama 3.2 LLM
Natural language insights and recommendations
Context-aware analysis based on filtered data


Data Management

CSV file upload functionality
Real-time data filtering by fiscal year and department
Download filtered data for offline analysis
Automatic data cleaning and preparation



Technology Stack

Frontend: Streamlit
Data Processing: Pandas, NumPy
Visualization: Plotly Express, Plotly Graph Objects
AI/ML: Transformers, PyTorch, Llama 3.2
Development: Python 3.x

Getting Started

Clone the Repository
``` git clone https://github.com/yourusername/BudgetLens-AI.git
```cd BudgetLens-AI

Install Dependencies
``` pip install -r requirements.txt

Run the Application
``` streamlit run app.py


Input Data Format
The dashboard expects a CSV file with the following columns:

DEPT_NM: Department Name (string)
BUD: Budget Amount (numeric)
FY: Fiscal Year (string/numeric)

Features in Detail
1. Data Visualization

Trend Analysis: Track budget evolution over time with interactive line charts that show trends for each department
Distribution Views: Understand budget allocation using interactive pie charts with hover details
Heat Map Analysis: Identify patterns and anomalies in budget distribution across departments and years
Comparative Analysis: Compare departmental budgets using horizontal bar charts

2. AI Integration

Automated Insights: Generate AI-powered analysis of budget trends and patterns
Smart Recommendations: Receive context-aware suggestions based on current data filters
Natural Language Understanding: Convert complex budget data into easily understandable insights

3. User Interface

Clean Dashboard Layout: Intuitive interface with clear navigation
Interactive Filters: Dynamic filtering by fiscal year and department
Responsive Design: Optimized for both desktop and tablet viewing
Download Capability: Export filtered data in CSV format

Best Practices

Regularly update the input data for accurate analysis
Use consistent department names and fiscal year formats
Review AI insights alongside traditional visualizations for comprehensive understanding
Export and save important analyses for reporting purposes

Future Enhancements

Advanced anomaly detection
Predictive budget forecasting
Custom report generation
API integration for automated data updates
Multi-user support with role-based access

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.


