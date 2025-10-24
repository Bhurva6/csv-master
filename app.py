import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configure the page
st.set_page_config(
    page_title="CSV Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("📊 CSV Operations")
st.markdown("---")

# Features section - Always visible
st.header("✨ Features")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("👨‍💼 Create Personas", use_container_width=True):
        st.session_state['feature'] = 'personas'
    st.markdown("""
    Generate detailed customer personas from your data
    """)

with col2:
    if st.button("📧 Verify Emails", use_container_width=True):
        st.session_state['feature'] = 'emails'
    st.markdown("""
    Advanced email validation with legitimacy scoring & Excel export
    """)

with col3:
    if st.button("📞 Verify Phone Numbers", use_container_width=True):
        st.session_state['feature'] = 'phones'
    st.markdown("""
    Advanced phone validation with format checking, region detection & Excel export
    """)

st.markdown("---")

# Sidebar for file upload
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file to explore and visualize your data"
)

# Check for uploaded file or sample data
if uploaded_file is not None or 'uploaded_file' in st.session_state:
    # Handle both real uploads and sample data
    if uploaded_file is not None:
        file_data = uploaded_file
        filename = uploaded_file.name
    else:
        # Use sample data from session state
        file_data = st.session_state['uploaded_file']
        filename = st.session_state.get('uploaded_filename', 'sample_data.csv')
    
    try:
        # Read the CSV file
        if isinstance(file_data, bytes):
            import io
            df = pd.read_csv(io.BytesIO(file_data))
        else:
            df = pd.read_csv(file_data)
        
        # Store the DataFrame in session state for use in features
        st.session_state['df'] = df
        
        # Data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Column information
        st.subheader("📊 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Data visualization section
        st.header("📈 Data Visualization")
        
        # Select columns for visualization
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_columns:
            st.subheader("Numeric Data Analysis")
            
            # Statistical summary
            st.write("**Statistical Summary:**")
            st.dataframe(df[numeric_columns].describe(), use_container_width=True)
            
            # Correlation heatmap
            if len(numeric_columns) > 1:
                st.write("**Correlation Heatmap:**")
                correlation_matrix = df[numeric_columns].corr()
                fig_heatmap = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Distribution plots
            st.write("**Distribution Plots:**")
            selected_numeric = st.selectbox("Select a numeric column:", numeric_columns)
            if selected_numeric:
                fig_hist = px.histogram(
                    df, 
                    x=selected_numeric, 
                    title=f"Distribution of {selected_numeric}",
                    marginal="box"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        if categorical_columns:
            st.subheader("Categorical Data Analysis")
            selected_categorical = st.selectbox("Select a categorical column:", categorical_columns)
            if selected_categorical:
                # Value counts
                value_counts = df[selected_categorical].value_counts().head(20)
                
                col1, col2 = st.columns(2)
                with col1:
                    # Bar chart
                    fig_bar = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top Values in {selected_categorical}",
                        labels={'x': selected_categorical, 'y': 'Count'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig_pie = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution of {selected_categorical}"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        # Custom visualization
        if len(numeric_columns) >= 2:
            st.subheader("Custom Scatter Plot")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_axis = st.selectbox("X-axis:", numeric_columns, key="x_axis")
            with col2:
                y_axis = st.selectbox("Y-axis:", numeric_columns, key="y_axis", index=1 if len(numeric_columns) > 1 else 0)
            with col3:
                color_by = st.selectbox("Color by:", ["None"] + categorical_columns, key="color_by")
            
            if x_axis and y_axis:
                fig_scatter = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=color_by if color_by != "None" else None,
                    title=f"{y_axis} vs {x_axis}",
                    hover_data=df.columns[:5].tolist()  # Show first 5 columns on hover
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Data filtering
        st.header("🔍 Data Filtering")
        st.write("Filter your data based on column values:")
        
        filters = {}
        for col in df.columns:
            if df[col].dtype in ['object']:
                unique_values = df[col].dropna().unique()
                if len(unique_values) <= 20:  # Only show filter for columns with <= 20 unique values
                    selected_values = st.multiselect(
                        f"Filter by {col}:",
                        options=unique_values,
                        key=f"filter_{col}"
                    )
                    if selected_values:
                        filters[col] = selected_values
            elif df[col].dtype in ['int64', 'float64']:
                min_val, max_val = float(df[col].min()), float(df[col].max())
                if min_val != max_val:
                    range_values = st.slider(
                        f"Filter {col} range:",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"range_{col}"
                    )
                    if range_values != (min_val, max_val):
                        filters[col] = range_values
        
        # Apply filters
        filtered_df = df.copy()
        for col, values in filters.items():
            if df[col].dtype in ['object']:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
            else:
                filtered_df = filtered_df[
                    (filtered_df[col] >= values[0]) & 
                    (filtered_df[col] <= values[1])
                ]
        
        if len(filters) > 0:
            st.write(f"**Filtered Data ({len(filtered_df)} rows)**")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error reading the CSV file: {str(e)}")
        st.info("Please make sure your file is a valid CSV format.")

else:
    # Landing page when no file is uploaded
    st.info("👈 Please upload a CSV file using the sidebar to get started!")
    
    # Sample data section
    st.header("🎯 Try with Sample Data")
    
    if st.button("Generate Sample Data"):
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'Name': [f'Person_{i}' for i in range(100)],
            'Age': np.random.randint(18, 65, 100),
            'Salary': np.random.normal(50000, 15000, 100),
            'Department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 100),
            'Experience': np.random.randint(0, 20, 100),
            'Performance_Score': np.random.uniform(1, 5, 100),
            'Gender': np.random.choice(['Male', 'Female', 'Non-binary', 'Prefer not to say'], 100),
            'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], 100),
            'Email': [
                f"{f'Person_{i}'.lower().replace(' ', '.')}.{i}@example.com" if i % 10 != 0 else 
                f"user{i}@gmail.com" if i % 10 == 1 else
                f"contact{i}@yahoo.com" if i % 10 == 2 else
                f"info{i}@company.com" if i % 10 == 3 else
                f"test{i}@temp-mail.org" if i % 10 == 4 else
                f"person{i}@outlook.com" if i % 10 == 5 else
                f"employee{i}@business.net" if i % 10 == 6 else
                f"client{i}@mailinator.com" if i % 10 == 7 else
                "" if i % 10 == 8 else  # Empty email for testing
                f"invalid-email{i}"  # Invalid format for testing
                for i in range(100)
            ],
            'Phone': [
                f"+1-{np.random.randint(200,999)}-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}" if i % 15 != 0 else
                f"({np.random.randint(200,999)}) {np.random.randint(100,999)}-{np.random.randint(1000,9999)}" if i % 15 == 1 else
                f"+44 {np.random.randint(20,99)} {np.random.randint(1000,9999)} {np.random.randint(1000,9999)}" if i % 15 == 2 else
                f"+91 {np.random.randint(7000000000,9999999999)}" if i % 15 == 3 else
                f"{np.random.randint(200,999)}-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}" if i % 15 == 4 else
                f"+1-{np.random.randint(200,999)}-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}, +44 {np.random.randint(20,99)} {np.random.randint(1000,9999)} {np.random.randint(1000,9999)}" if i % 15 == 5 else  # Multiple phones
                f"({np.random.randint(200,999)}) {np.random.randint(100,999)}-{np.random.randint(1000,9999)}; +91 {np.random.randint(7000000000,9999999999)}" if i % 15 == 6 else  # Multiple phones
                f"invalid-phone-{i}" if i % 15 == 7 else  # Invalid format
                f"123" if i % 15 == 8 else  # Too short
                "" if i % 15 == 9 else  # Empty
                f"+{np.random.randint(100,999)} {np.random.randint(100000,999999)}"  # International format
                for i in range(100)
            ]
        })
        
        st.success("Sample data generated!")
        st.dataframe(sample_data.head(10), use_container_width=True)
        
        # Store sample data in session state for immediate use
        st.session_state['sample_data'] = sample_data
        
        # Download sample data
        csv_sample = sample_data.to_csv(index=False)
        st.download_button(
            label="Download sample data",
            data=csv_sample,
            file_name="sample_data.csv",
            mime="text/csv"
        )
        
        # Button to use sample data immediately
        if st.button("Use This Sample Data Now"):
            # Convert to CSV bytes and simulate upload
            csv_bytes = sample_data.to_csv(index=False).encode()
            st.session_state['uploaded_file'] = csv_bytes
            st.session_state['uploaded_filename'] = 'sample_data.csv'
            st.rerun()

# Feature implementation - Always available when feature is selected
if 'feature' in st.session_state:
    st.markdown("---")

    if st.session_state['feature'] == 'personas':
        st.header("👨‍💼 Create Personas")

        # Persona creation logic
        if 'df' in st.session_state:
            try:
                # Use the DataFrame from session state
                df = st.session_state['df']

                # Check for relevant columns - expanded detection
                name_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['name', 'full_name', 'first_name', 'last_name', 'customer_name', 'person_name', 'user_name'])]
                age_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['age', 'age_years', 'person_age', 'user_age', 'years_old'])]
                gender_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['gender', 'sex', 'male', 'female', 'person_gender', 'user_gender'])]
                location_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['city', 'state', 'country', 'location', 'address', 'region', 'province', 'town', 'place'])]

                if name_cols or age_cols or gender_cols or location_cols:
                    found_cols = []
                    if name_cols:
                        found_cols.append(f"Name: {', '.join(name_cols)}")
                    if age_cols:
                        found_cols.append(f"Age: {', '.join(age_cols)}")
                    if gender_cols:
                        found_cols.append(f"Gender: {', '.join(gender_cols)}")
                    if location_cols:
                        found_cols.append(f"Location: {', '.join(location_cols)}")
                    
                    st.success(f"Found relevant columns for persona creation: {', '.join(found_cols)}")

                    # Allow user to select number of personas
                    max_personas = min(10, len(df))  # Limit to 10 or available rows
                    num_personas = st.slider("Number of personas to generate:", min_value=1, max_value=max_personas, value=min(5, max_personas))

                    # Generate personas from random sample
                    if len(df) > num_personas:
                        # Randomly sample rows for diversity
                        sample_indices = np.random.choice(len(df), size=num_personas, replace=False)
                    else:
                        sample_indices = range(len(df))

                    # Generate personas
                    personas = []
                    for i, idx in enumerate(sample_indices):
                        persona = {
                            'Name': f"Persona {i+1}",
                            'Age': "Unknown",
                            'Gender': "Unknown", 
                            'Location': "Unknown"
                        }
                        
                        if name_cols:
                            persona['Name'] = df[name_cols[0]].iloc[idx] if pd.notna(df[name_cols[0]].iloc[idx]) else f"Persona {i+1}"
                        
                        if age_cols:
                            persona['Age'] = df[age_cols[0]].iloc[idx] if pd.notna(df[age_cols[0]].iloc[idx]) else "Unknown"
                        
                        if gender_cols:
                            persona['Gender'] = df[gender_cols[0]].iloc[idx] if pd.notna(df[gender_cols[0]].iloc[idx]) else "Unknown"
                        
                        if location_cols:
                            persona['Location'] = df[location_cols[0]].iloc[idx] if pd.notna(df[location_cols[0]].iloc[idx]) else "Unknown"

                        personas.append(persona)

                    # Display personas
                    for i, persona in enumerate(personas):
                        with st.expander(f"**{persona['Name']}**"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Age:** {persona['Age']}")
                                st.write(f"**Gender:** {persona['Gender']}")
                            with col2:
                                st.write(f"**Location:** {persona['Location']}")
                                st.write("**Characteristics:** Based on data analysis")

                    # Download personas
                    persona_df = pd.DataFrame(personas)
                    csv_personas = persona_df.to_csv(index=False)
                    st.download_button(
                        label="Download Personas as CSV",
                        data=csv_personas,
                        file_name="personas.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No relevant columns found for persona creation. Please ensure your CSV has columns containing keywords like 'name', 'age', 'gender', 'sex', 'city', 'state', 'country', or 'location'.")
                    st.info(f"Available columns in your CSV: {', '.join(df.columns.tolist())}")
                    st.info("If your columns have different names, you may need to rename them or contact support.")

            except Exception as e:
                st.error(f"Error processing data for personas: {str(e)}")
        else:
            st.info("Please upload a CSV file first to create personas.")

    elif st.session_state['feature'] == 'emails':
        st.header("📧 Verify Emails")

        # Email verification logic - works with uploaded CSV or sample data
        if 'df' in st.session_state:
            try:
                # Use the DataFrame from session state
                df = st.session_state['df']

                # Find email columns
                email_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['email', 'mail', 'e-mail', 'email_address'])]

                if email_cols:
                    st.success(f"Found email column(s): {', '.join(email_cols)}")

                    # Select which email column to verify
                    selected_email_col = st.selectbox("Select email column to verify:", email_cols)

                    if st.button("Verify Emails in CSV"):
                        with st.spinner("Verifying emails..."):
                            import re

                            # Enhanced email validation
                            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

                            # Common disposable/temporary email domains
                            disposable_domains = [
                                '10minutemail.com', 'guerrillamail.com', 'mailinator.com',
                                'temp-mail.org', 'throwaway.email', 'yopmail.com',
                                'maildrop.cc', 'tempail.com', 'dispostable.com'
                            ]

                            results = []
                            for idx, email in enumerate(df[selected_email_col]):
                                if pd.isna(email) or str(email).strip() == '':
                                    results.append({
                                        'Row': idx + 1,
                                        'Email': email if pd.notna(email) else '',
                                        'Status': 'Empty',
                                        'Notes': 'Missing email address'
                                    })
                                    continue

                                email = str(email).strip().lower()
                                is_valid_format = bool(re.match(email_pattern, email))

                                if is_valid_format:
                                    domain = email.split('@')[1]
                                    is_disposable = domain in disposable_domains
                                    is_deliverable = not any(x in email for x in ['test', 'example', 'invalid', 'fake'])

                                    if is_deliverable and not is_disposable:
                                        status = 'Valid (Deliverable)'
                                        notes = f'Valid format, {domain} domain'
                                    elif is_disposable:
                                        status = 'Invalid (Disposable)'
                                        notes = f'Disposable email domain: {domain}'
                                    else:
                                        status = 'Invalid (Undeliverable)'
                                        notes = f'Potentially undeliverable: {domain}'
                                else:
                                    status = 'Invalid (Format)'
                                    notes = 'Invalid email format'

                                results.append({
                                    'Row': idx + 1,
                                    'Email': email,
                                    'Status': status,
                                    'Notes': notes
                                })

                            results_df = pd.DataFrame(results)

                            # Summary metrics
                            st.subheader("📊 Email Verification Summary")
                            
                            total_emails = len(results)
                            valid_deliverable = sum(1 for r in results if r['Status'] == 'Valid (Deliverable)')
                            invalid_emails = sum(1 for r in results if r['Status'].startswith('Invalid'))
                            empty_emails = sum(1 for r in results if r['Status'] == 'Empty')

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Emails", total_emails)
                            with col2:
                                st.metric("Valid (Deliverable)", valid_deliverable)
                            with col3:
                                st.metric("Invalid Emails", invalid_emails)
                            with col4:
                                st.metric("Empty Emails", empty_emails)

                            # Success rate
                            if total_emails > 0:
                                success_rate = (valid_deliverable / total_emails) * 100
                                st.success(f"✅ **{success_rate:.1f}%** of emails are valid and deliverable!")

                            # Visual representation
                            status_counts = results_df['Status'].value_counts()
                            fig_pie = px.pie(
                                values=status_counts.values,
                                names=status_counts.index,
                                title="Email Verification Status Distribution",
                                color_discrete_sequence=['#00ff00', '#ff0000', '#ffff00', '#ff9800']
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                            # Show results table
                            st.subheader("📋 Detailed Results")
                            st.dataframe(results_df, use_container_width=True)

                            # Create Excel file for download
                            from io import BytesIO
                            import openpyxl
                            from openpyxl.styles import PatternFill, Font

                            wb = openpyxl.Workbook()
                            ws = wb.active
                            ws.title = "Email Verification Results"

                            # Add summary at the top
                            ws['A1'] = "Email Verification Summary"
                            ws['A1'].font = Font(bold=True, size=14)
                            
                            ws['A3'] = "Total Emails:"
                            ws['B3'] = total_emails
                            ws['A4'] = "Valid (Deliverable):"
                            ws['B4'] = valid_deliverable
                            ws['A5'] = "Invalid Emails:"
                            ws['B5'] = invalid_emails
                            ws['A6'] = "Empty Emails:"
                            ws['B6'] = empty_emails
                            ws['A7'] = f"Success Rate: {success_rate:.1f}%" if total_emails > 0 else "Success Rate: N/A"

                            # Add detailed results starting from row 10
                            start_row = 10
                            headers = list(results_df.columns)
                            for col_num, header in enumerate(headers, 1):
                                cell = ws.cell(row=start_row, column=col_num, value=header)
                                cell.font = Font(bold=True)
                                cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")

                            # Add data with conditional formatting
                            for row_num, row_data in enumerate(results_df.values, start_row + 1):
                                for col_num, value in enumerate(row_data, 1):
                                    cell = ws.cell(row=row_num, column=col_num, value=str(value))

                                    # Color coding based on status
                                    if col_num == 3:  # Status column
                                        if 'Valid' in str(value):
                                            cell.fill = PatternFill(start_color="C8E6C9", end_color="C8E6C9", fill_type="solid")
                                        elif 'Invalid' in str(value):
                                            cell.fill = PatternFill(start_color="FFCDD2", end_color="FFCDD2", fill_type="solid")
                                        elif 'Empty' in str(value):
                                            cell.fill = PatternFill(start_color="FFF9C4", end_color="FFF9C4", fill_type="solid")

                            # Auto-adjust column widths
                            for column in ws.columns:
                                max_length = 0
                                column_letter = column[0].column_letter
                                for cell in column:
                                    try:
                                        if len(str(cell.value)) > max_length:
                                            max_length = len(str(cell.value))
                                    except:
                                        pass
                                adjusted_width = min(max_length + 2, 50)
                                ws.column_dimensions[column_letter].width = adjusted_width

                            # Save to BytesIO
                            excel_buffer = BytesIO()
                            wb.save(excel_buffer)
                            excel_buffer.seek(0)

                            # Download button
                            st.download_button(
                                label="📥 Download Verification Results (Excel)",
                                data=excel_buffer,
                                file_name="email_verification_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                else:
                    st.warning("No email columns found in your CSV. Please ensure your file has columns containing 'email', 'mail', 'e-mail', or 'email_address' in the name.")
                    st.info(f"Available columns in your CSV: {', '.join(df.columns.tolist())}")
                    st.info("If your email column has a different name, you may need to rename it or contact support.")

            except Exception as e:
                st.error(f"Error processing CSV for email verification: {str(e)}")
        else:
            st.info("Please upload a CSV file first to verify emails.")

    elif st.session_state['feature'] == 'phones':
        st.header("📞 Verify Phone Numbers")

        # Phone verification logic - works with uploaded CSV or sample data
        if 'df' in st.session_state:
            try:
                # Use the DataFrame from session state
                df = st.session_state['df']

                # Find phone columns
                phone_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['phone', 'mobile', 'tel', 'contact', 'cell', 'telephone', 'number'])]

                if phone_cols:
                    st.success(f"Found phone column(s): {', '.join(phone_cols)}")

                    # Select which phone column to verify
                    selected_phone_col = st.selectbox("Select phone column to verify:", phone_cols)

                    if st.button("Verify Phone Numbers in CSV"):
                        with st.spinner("Verifying phone numbers..."):
                            import re

                            results = []
                            for idx, phone_data in enumerate(df[selected_phone_col]):
                                if pd.isna(phone_data) or str(phone_data).strip() == '':
                                    results.append({
                                        'Row': idx + 1,
                                        'Original Data': phone_data if pd.notna(phone_data) else '',
                                        'Phone Number': '',
                                        'Status': 'Empty',
                                        'Notes': 'Missing phone number'
                                    })
                                    continue

                                # Handle multiple phone numbers in a single cell
                                phone_string = str(phone_data).strip()
                                
                                # Split by common delimiters: comma, semicolon, newline, pipe, slash
                                phones_in_cell = re.split(r'[;,|\n/]+', phone_string)
                                phones_in_cell = [p.strip() for p in phones_in_cell if p.strip()]

                                if not phones_in_cell:
                                    results.append({
                                        'Row': idx + 1,
                                        'Original Data': phone_string,
                                        'Phone Number': '',
                                        'Status': 'Empty',
                                        'Notes': 'No phone numbers found after parsing'
                                    })
                                    continue

                                # Process each phone number in the cell
                                for phone_idx, phone in enumerate(phones_in_cell):
                                    # Remove all non-digit characters except + for international codes
                                    clean_phone = re.sub(r'[^\d+]', '', phone)

                                    # Basic validation patterns
                                    us_pattern = r'^\+?1?\d{10}$'
                                    international_pattern = r'^\+\d{1,4}\d{6,14}$'

                                    is_us_format = bool(re.match(us_pattern, clean_phone))
                                    is_international = bool(re.match(international_pattern, clean_phone))

                                    # Determine country/region
                                    if clean_phone.startswith('+1') or (len(clean_phone) == 10 and not clean_phone.startswith('+')):
                                        region = "US/Canada"
                                    elif clean_phone.startswith('+44'):
                                        region = "UK"
                                    elif clean_phone.startswith('+91'):
                                        region = "India"
                                    elif clean_phone.startswith('+61'):
                                        region = "Australia"
                                    elif clean_phone.startswith('+86'):
                                        region = "China"
                                    elif clean_phone.startswith('+81'):
                                        region = "Japan"
                                    elif clean_phone.startswith('+49'):
                                        region = "Germany"
                                    elif clean_phone.startswith('+33'):
                                        region = "France"
                                    elif clean_phone.startswith('+'):
                                        region = "International"
                                    else:
                                        region = "Unknown"

                                    is_valid = is_us_format or is_international

                                    results.append({
                                        'Row': idx + 1,
                                        'Original Data': phone_string if phone_idx == 0 else '',
                                        'Phone Number': phone,
                                        'Cleaned': clean_phone,
                                        'Format Valid': '✅ Valid' if is_valid else '❌ Invalid',
                                        'Region': region,
                                        'Status': 'Valid' if is_valid else 'Invalid',
                                        'Notes': f'Phone {phone_idx + 1} of {len(phones_in_cell)} in cell' if len(phones_in_cell) > 1 else ''
                                    })

                            results_df = pd.DataFrame(results)

                            # Summary metrics
                            st.subheader("📊 Phone Verification Summary")
                            
                            total_phones = len(results)
                            valid_phones = sum(1 for r in results if r['Status'] == 'Valid')
                            invalid_phones = sum(1 for r in results if r['Status'] == 'Invalid')
                            empty_entries = sum(1 for r in results if r['Status'] == 'Empty')

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Phone Numbers", total_phones)
                            with col2:
                                st.metric("Valid Numbers", valid_phones)
                            with col3:
                                st.metric("Invalid Numbers", invalid_phones)
                            with col4:
                                st.metric("Empty Entries", empty_entries)

                            # Success rate
                            if total_phones > 0:
                                success_rate = (valid_phones / total_phones) * 100
                                st.success(f"✅ **{success_rate:.1f}%** of phone numbers are valid!")

                            # Visual representation
                            status_counts = results_df['Status'].value_counts()
                            fig_pie = px.pie(
                                values=status_counts.values,
                                names=status_counts.index,
                                title="Phone Verification Status Distribution",
                                color_discrete_sequence=['#00ff00', '#ff0000', '#ffff00', '#ff9800']
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                            # Show results table
                            st.subheader("📋 Detailed Results")
                            st.dataframe(results_df, use_container_width=True)

                            # Create Excel file for download
                            from io import BytesIO
                            import openpyxl
                            from openpyxl.styles import PatternFill, Font

                            wb = openpyxl.Workbook()
                            ws = wb.active
                            ws.title = "Phone Verification Results"

                            # Add summary at the top
                            ws['A1'] = "Phone Verification Summary"
                            ws['A1'].font = Font(bold=True, size=14)
                            
                            ws['A3'] = "Total Phone Numbers:"
                            ws['B3'] = total_phones
                            ws['A4'] = "Valid Numbers:"
                            ws['B4'] = valid_phones
                            ws['A5'] = "Invalid Numbers:"
                            ws['B5'] = invalid_phones
                            ws['A6'] = "Empty Entries:"
                            ws['B6'] = empty_entries
                            ws['A7'] = f"Success Rate: {success_rate:.1f}%" if total_phones > 0 else "Success Rate: N/A"

                            # Add detailed results starting from row 10
                            start_row = 10
                            headers = list(results_df.columns)
                            for col_num, header in enumerate(headers, 1):
                                cell = ws.cell(row=start_row, column=col_num, value=header)
                                cell.font = Font(bold=True)
                                cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")

                            # Add data with conditional formatting
                            for row_num, row_data in enumerate(results_df.values, start_row + 1):
                                for col_num, value in enumerate(row_data, 1):
                                    cell = ws.cell(row=row_num, column=col_num, value=str(value))

                                    # Color coding based on status
                                    if col_num == 6:  # Status column
                                        if 'Valid' in str(value):
                                            cell.fill = PatternFill(start_color="C8E6C9", end_color="C8E6C9", fill_type="solid")
                                        elif 'Invalid' in str(value):
                                            cell.fill = PatternFill(start_color="FFCDD2", end_color="FFCDD2", fill_type="solid")
                                        elif 'Empty' in str(value):
                                            cell.fill = PatternFill(start_color="FFF9C4", end_color="FFF9C4", fill_type="solid")

                            # Auto-adjust column widths
                            for column in ws.columns:
                                max_length = 0
                                column_letter = column[0].column_letter
                                for cell in column:
                                    try:
                                        if len(str(cell.value)) > max_length:
                                            max_length = len(str(cell.value))
                                    except:
                                        pass
                                adjusted_width = min(max_length + 2, 50)
                                ws.column_dimensions[column_letter].width = adjusted_width

                            # Save to BytesIO
                            excel_buffer = BytesIO()
                            wb.save(excel_buffer)
                            excel_buffer.seek(0)

                            # Download button
                            st.download_button(
                                label="📥 Download Verification Results (Excel)",
                                data=excel_buffer,
                                file_name="phone_verification_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                else:
                    st.warning("No phone columns found in your CSV. Please ensure your file has columns containing 'phone', 'mobile', 'tel', 'contact', 'cell', 'telephone', or 'number' in the name.")
                    st.info(f"Available columns in your CSV: {', '.join(df.columns.tolist())}")
                    st.info("If your phone column has a different name, you may need to rename it or contact support.")

            except Exception as e:
                st.error(f"Error processing CSV for phone verification: {str(e)}")
        else:
            st.info("Please upload a CSV file first to verify phone numbers.")
    
    # Clear feature button
    if st.button("🔄 Back to Features"):
        del st.session_state['feature']
