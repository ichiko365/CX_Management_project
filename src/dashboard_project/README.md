# Support Dashboard

A Streamlit dashboard for customer support management showing urgent feedback queue and team performance metrics.

## Features

- **Urgent Feedback Queue**: Shows the latest customer complaints/issues
- **Team Performance**: Displays task assignment and completion rates for team members
- **Data Synchronization**: Sync complaints from customer database to main database
- **Cache Management**: Refresh data and clear cache to avoid stale connections

## Setup

1. **Install Dependencies**:
   ```bash
   pip install streamlit pandas psycopg2-binary
   ```

2. **Database Configuration**: 
   Ensure your Streamlit secrets are configured with database connections:
   ```toml
   # .streamlit/secrets.toml
   [connections.postgresql]
   user = "your_username"
   password = "your_password" 
   host = "your_host"
   port = 5432
   database = "main_database_name"
   customer_database = "customer_database_name"
   ```

3. **Images (Optional)**:
   Place PNG/JPEG images in `assets/` folder:
   - `assets/urgent_queue.png` - for urgent queue card
   - `assets/team_performance.png` - for team performance card

## Running the Dashboard

```bash
# Navigate to dashboard directory
cd src/dashboard_project

# Run with Streamlit
streamlit run dashboard.py
```

The dashboard will open in your browser at http://localhost:8501

## Usage

1. **Sync Data**: Click "Sync complaints from customer DB" to fetch latest data from customer database
2. **Refresh**: Click "Refresh Data" to clear cache and reload data if you encounter connection issues
3. **View Data**: Browse the urgent feedback queue and team performance metrics

## Troubleshooting

### Connection Issues
- If you see "connection already closed" errors, click "Refresh Data" button
- Check your database credentials in `.streamlit/secrets.toml`
- Ensure both databases are accessible

### Missing Data
- Use "Sync complaints from customer DB" button to populate data
- Check that source tables exist: `support_tasks`, `team_members`, `departments`, `users`

### Image Issues  
- Images should be valid PNG/JPEG files
- Place them in the `assets/` directory
- Dashboard will show placeholder text if images are missing/invalid

## Files Structure

```
src/dashboard_project/
├── dashboard.py           # Main Streamlit app
├── database/
│   ├── queries.py        # Database query functions
│   └── connector.py      # Database connection utilities
├── assets/               # Image assets
│   ├── urgent_queue.png
│   └── team_performance.png
└── README.md            # This file
```

## Notes

- Priority and Rating columns are automatically excluded from display
- The dashboard uses fallback sample data if database is unavailable
- Connection caching is optimized to avoid "connection closed" errors
- Data is refreshed automatically but you can force refresh using the button
