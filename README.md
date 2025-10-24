# CSV Data Explorer 📊

A powerful Streamlit application for exploring and visualizing CSV data with interactive charts and filtering capabilities.

## Features

- **Data Upload**: Easy CSV file upload with drag-and-drop support
- **Data Overview**: Key statistics, column information, and data preview
- **Email Verification**: Advanced email validation with legitimacy scoring and Excel export
- **Phone Number Verification**: Validate phone formats and identify regions
- **Persona Creation**: Generate customer personas from demographic data
- **Visualizations**: 
  - Correlation heatmaps
  - Distribution plots
  - Bar charts and pie charts
  - Interactive scatter plots
- **Data Filtering**: Real-time filtering by categorical and numeric columns
- **Export**: Download filtered data, verification results, and personas

## Setup Instructions

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Upload Data**: Use the sidebar to upload your CSV file
2. **Explore**: View data overview, statistics, and column information
3. **Visualize**: Explore different charts and visualizations
4. **Filter**: Use the filtering options to focus on specific data
5. **Export**: Download filtered results

## Sample Data

The app includes a "Generate Sample Data" feature that creates a dataset with 100 rows containing:
- **Personal Information**: Names, ages, departments
- **Professional Data**: Salaries, experience, performance scores  
- **Contact Information**: Email addresses with various formats for testing

### Testing Features

Use the sample data to test all features:
- **Email Verification**: Includes valid, invalid, empty, and disposable email domains
- **Persona Creation**: Contains demographic data for generating customer profiles
- **Data Analysis**: Rich dataset with numeric and categorical columns

Click "Use This Sample Data Now" to immediately load the generated data without downloading/uploading.

## Requirements

- Python 3.7+
- Streamlit 1.28.0+
- Pandas 2.0.0+
- NumPy 1.24.0+
- Plotly 5.15.0+

## Project Structure

```
csv-master/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Contributing

Feel free to contribute to this project by:
- Adding new visualization types
- Improving the UI/UX
- Adding data export formats
- Enhancing filtering capabilities

## License

This project is open source and available under the MIT License.
