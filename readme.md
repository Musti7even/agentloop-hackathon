# AgentLoop Hackathon

## Python Environment Setup

### Creating a Virtual Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Verify activation:**
   ```bash
   which python  # Should point to venv/bin/python
   ```

### Managing Dependencies

1. **Install packages:**
   ```bash
   python3 -m pip install package_name
   ```

2. **Install from requirements.txt:**
   ```bash
   python3 -m pip install -r requirements.txt
   ```