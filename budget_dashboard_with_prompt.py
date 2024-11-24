import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

@st.cache_resource
def load_llm():
    """Load and cache the LLM model and tokenizer"""
    with st.spinner('Loading Llama model...'):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        pipe = pipeline("text-generation", 
                       model="meta-llama/Llama-3.2-1B", 
                       device='mps' if torch.backends.mps.is_available() else 'cpu')
    return pipe, tokenizer, model
def clean_and_prepare_data(df):
    """Clean and prepare the dataframe"""
    # Convert BUD to numeric
    df['BUD'] = pd.to_numeric(df['BUD'], errors='coerce')
    
    # Convert DEPT_NM to string
    df['DEPT_NM'] = df['DEPT_NM'].astype(str)
    
    # Convert FY to string
    df['FY'] = df['FY'].astype(str)
    
    # Remove any rows where BUD is NaN
    df = df.dropna(subset=['BUD'])
    
    return df

def create_pie_chart(data):
    chart_data = data.groupby('DEPT_NM')['BUD'].sum().reset_index()
    
    fig = px.pie(
        chart_data,
        values="BUD",
        names="DEPT_NM",
        title="Budget Distribution by Department",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hole=0.4
    )
    
    fig.update_layout(
        title_x=0.5,
        legend={'orientation': 'h', 'y': -0.1}
    )
    
    return fig

def create_heatmap(data):
    # Pivot the data for heatmap
    heatmap_data = data.pivot_table(
        values='BUD',
        index='DEPT_NM',
        columns='FY',
        aggfunc='sum'
    ).fillna(0)
    
    # Normalize the data for better visualization
    normalized_data = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=normalized_data.values,
        x=normalized_data.columns,
        y=normalized_data.index,
        colorscale='Blues',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Budget Intensity by Department and Year',
        title_x=0.5,
        xaxis_title='Fiscal Year',
        yaxis_title='Department',
        height=600
    )
    
    return fig

def create_trend_line(data):
    trend_data = data.groupby(['FY', 'DEPT_NM'])['BUD'].sum().reset_index()
    
    fig = px.line(
        trend_data,
        x='FY',
        y='BUD',
        color='DEPT_NM',
        title='Budget Trends Over Time',
        markers=True
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title='Fiscal Year',
        yaxis_title='Budget Amount',
        legend={'orientation': 'h', 'y': -0.2}
    )
    
    return fig

def create_bar_chart(data):
    dept_totals = data.groupby('DEPT_NM')['BUD'].sum().reset_index()
    dept_totals = dept_totals.sort_values('BUD', ascending=True)
    
    fig = px.bar(
        dept_totals,
        x='BUD',
        y='DEPT_NM',
        orientation='h',
        title='Budget Distribution by Department'
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title='Budget Amount',
        yaxis_title='Department'
    )
    
    return fig


def generate_default_prompt(total_budget, avg_budget, selected_depts, selected_years, filtered_df):
    """Generate a default prompt template with budget insights"""
    dept_budgets = filtered_df.groupby('DEPT_NM')['BUD'].sum().to_dict()
    dept_insights = "\n".join([f"{dept}: ${budget:,.2f}" for dept, budget in dept_budgets.items()])
    
    return f"""Analyze this budget data summary:
Total Budget: ${total_budget:,.2f}
Average Budget: ${avg_budget:,.2f}
Number of Departments: {len(selected_depts)}
Years covered: {', '.join(selected_years)}

Department-wise breakdown:
{dept_insights}

Please provide:
1. Key trends and patterns
2. Notable insights
3. Potential areas of optimization
4. Budget allocation recommendations
"""

def main():
    st.set_page_config(layout="wide", page_title="Budget Analysis Dashboard")
    
    st.title("Budget Analysis Dashboard")
    st.markdown("---")
    
    # Load LLM model
    pipe, tokenizer, model = load_llm()
    
    uploaded_file = st.file_uploader("Upload your budget CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Clean and prepare data
            df = clean_and_prepare_data(df)
            
            # Show raw data
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
            
            # Filters
            st.sidebar.header("Filters")
            
            # Year filter - Convert to list for sorting
            years = sorted(list(df['FY'].unique()))
            selected_years = st.sidebar.multiselect(
                "Select Fiscal Years",
                options=years,
                default=years
            )
            
            # Department filter - Convert to list for sorting
            depts = sorted(list(df['DEPT_NM'].unique()))
            selected_depts = st.sidebar.multiselect(
                "Select Departments",
                options=depts,
                default=depts[:5] if len(depts) > 5 else depts
            )
            
            # Filter data
            filtered_df = df[
                (df['FY'].isin(selected_years)) &
                (df['DEPT_NM'].isin(selected_depts))
            ]
            
            if len(filtered_df) == 0:
                st.warning("No data available for the selected filters.")
                return
            
            # Display metrics
            total_budget = filtered_df['BUD'].sum()
            avg_budget = filtered_df['BUD'].mean()
            max_budget = filtered_df['BUD'].max()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Budget", f"${total_budget:,.2f}")
            col2.metric("Average Budget", f"${avg_budget:,.2f}")
            col3.metric("Max Budget", f"${max_budget:,.2f}")
            
            # Create tabs for visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Trend Analysis",
                "Distribution",
                "Heatmap",
                "Department Comparison",
                "AI Insights"
            ])
            
            with tab1:
                st.plotly_chart(create_trend_line(filtered_df), use_container_width=True)
            
            with tab2:
                st.plotly_chart(create_pie_chart(filtered_df), use_container_width=True)
            
            with tab3:
                st.plotly_chart(create_heatmap(filtered_df), use_container_width=True)
            
            with tab4:
                st.plotly_chart(create_bar_chart(filtered_df), use_container_width=True)
            
            with tab5:
                st.subheader("AI Budget Analysis")
                
                # Create columns for prompt options
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Generate default prompt
                    default_prompt = generate_default_prompt(
                        total_budget, 
                        avg_budget, 
                        selected_depts, 
                        selected_years,
                        filtered_df
                    )
                    
                    # Add prompt templates
                    prompt_template = st.selectbox(
                        "Select Analysis Type",
                        options=[
                            "Custom Analysis",
                            "Trend Analysis",
                            "Cost Optimization",
                            "Department Comparison",
                            "Risk Assessment",
                            "Future Recommendations"
                        ]
                    )
                    
                    # Template-specific additional context
                    template_prompts = {
                        "Trend Analysis": "\nFocus on identifying historical patterns and future projections in the budget data.",
                        "Cost Optimization": "\nIdentify potential areas for cost optimization and efficiency improvements.",
                        "Department Comparison": "\nCompare department performances and highlight significant variations.",
                        "Risk Assessment": "\nAnalyze potential budget risks and suggest mitigation strategies.",
                        "Future Recommendations": "\nProvide strategic recommendations for future budget planning."
                    }
                    
                    # Update prompt based on template
                    if prompt_template != "Custom Analysis":
                        default_prompt += template_prompts.get(prompt_template, "")
                    
                    # Custom prompt text area
                    user_prompt = st.text_area(
                        "Customize your analysis prompt",
                        value=default_prompt,
                        height=300,
                        help="Modify the prompt to focus on specific aspects of the budget data"
                    )
                
                with col2:
                    # Analysis parameters
                    max_length = st.slider(
                        "Response Length",
                        min_value=100,
                        max_value=500,
                        value=200,
                        step=50,
                        help="Adjust the length of the AI-generated response"
                    )
                    
                    temperature = st.slider(
                        "Creativity Level",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        help="Higher values make the output more creative but less focused"
                    )
                    
                # Generate insights button
                if st.button("Generate AI Insights", type="primary"):
                    with st.spinner("Analyzing budget data..."):
                        try:
                            # Generate insights using the LLM
                            response = pipe(
                                user_prompt,
                                max_length=max_length,
                                temperature=temperature,
                                num_return_sequences=1
                            )
                            
                            # Display insights in an expandable container
                            with st.expander("📊 Budget Insights", expanded=True):
                                st.markdown("### AI-Generated Analysis")
                                st.write(response[0]['generated_text'])
                                
                                # Add option to download the analysis
                                st.download_button(
                                    label="Download Analysis",
                                    data=f"Prompt:\n{user_prompt}\n\nAnalysis:\n{response[0]['generated_text']}",
                                    file_name="budget_analysis.txt",
                                    mime="text/plain"
                                )
                        
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
                            st.info("Please try adjusting the prompt or parameters and try again.")
                
                # Add help section
                with st.expander("ℹ️ How to Use AI Insights"):
                    st.markdown("""
                    ### Tips for Getting Better Insights:
                    1. **Choose Analysis Type**: Select a predefined template or use custom analysis
                    2. **Customize Prompt**: Modify the prompt to focus on specific aspects
                    3. **Adjust Parameters**:
                        - Increase response length for more detailed analysis
                        - Adjust creativity level based on your needs
                    4. **Iterate**: Try different prompts and parameters to get the most useful insights
                    """)
            
            # Add download button for filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name="filtered_budget_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("""
            Please ensure your CSV file contains these columns:
            - DEPT_NM (Department Name)
            - BUD (Budget Amount as number)
            - FY (Fiscal Year)
            
            To help debug, here are the first few rows of your data:
            """)
            if 'df' in locals():
                st.write(df.head())
                st.write("\nColumn types:")
                st.write(df.dtypes)
    
    else:
        st.info("👆 Please upload a CSV file to begin the analysis")

if __name__ == "__main__":
    main()