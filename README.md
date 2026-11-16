# SMARTCAI — Cairo Transportation & Urban Planning

SMARTCAI is a Python-based decision-support application for exploring transportation and urban-planning problems using sample road, traffic, neighborhood, facility, and public-transit data for Cairo, Egypt.

The application provides three interactive modules through a Streamlit interface:

- **Emergency Vehicle Routing** — explores routes between neighborhoods and facilities using graph-based pathfinding.
- **Urban Planning Optimization** — supports road-network planning, traffic-aware routing, and road-maintenance budget optimization.
- **Public Transit Optimization** — evaluates transit-route and fleet-allocation scenarios, including travel time and transfer penalties.

> **Project scope:** SMARTCAI is an academic/prototype decision-support project built around the datasets included in this repository. Its outputs should be treated as analytical estimates, not as live navigation, official transport schedules, or operational instructions.

## Features

### Emergency vehicle routing
- Models neighborhoods, facilities, and roads as a graph.
- Uses A* pathfinding to explore routes.
- Supports a priority-routing mode in the emergency-routing module.

### Urban planning
- Builds a road-network graph from the included data.
- Supports traffic-aware, time-dependent route analysis.
- Uses Kruskal-style network optimization with mandatory-connection validation.
- Includes greedy and dynamic-programming approaches to road-maintenance selection under a budget.

### Public transit
- Evaluates transit routes and travel-time scenarios.
- Accounts for transfer penalties when changing transit lines.
- Uses dynamic programming to compare route and fleet-allocation options.
- Includes logic for handling stops shared by multiple transit routes.

## Technology stack

- **Language:** Python
- **Application UI:** Streamlit
- **Graph algorithms:** NetworkX
- **Data processing:** pandas, NumPy
- **Visualization:** Matplotlib, PyVis, Folium, Streamlit-Folium, PyDeck

## Repository structure

The repository includes the Streamlit entry point, feature modules, algorithm implementations, utility functions, tests, and sample CSV datasets.

| File | Purpose |
| --- | --- |
| `app.py` | Main Streamlit application and module navigation |
| `emergency_routing.py` | Emergency-vehicle routing module |
| `urban_planning.py` | Road-network and urban-planning optimization |
| `public_transit.py` | Public-transit optimization interface |
| `algorithms.py` | Shared graph and optimization algorithms |
| `data.py` | Sample datasets defined in code (the app's actual data source) |
| `utils.py` | Shared utility functions |
| `tests.py` | Algorithm and optimization unit tests |
| `test_transit.py` | Transit and integration-oriented tests |
| `simulation.py` | Standalone vehicle-simulation prototype; not used by `app.py` and currently non-functional |
| `requirements.txt` | Pinned third-party dependencies |
| CSV files | Reference copies of sample road, traffic, neighborhood, facility, and transit data; not read by the application code |

## Getting started

### 1. Requirements

- Python 3.12 or newer (required by the pinned dependency versions; tested with Python 3.13).
- pip
- Git

### 2. Clone the repository

```bash
git clone https://github.com/Rewanooo22/SMARTCAI.git
cd SMARTCAI
```

### 3. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Git Bash):**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Run the application

From the project root:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Running the tests

The project tests use Python's built-in `unittest` framework.

Run both test modules from the project root:

```bash
python -m unittest tests test_transit -v
```

The test suite includes graph pathfinding, route and network optimization, maintenance-budget selection, and public-transit transfer behavior.

### Known data limitation

The Kruskal test for the mandatory hospital connections involving **F9** and **F10** is currently skipped because the corresponding road rows are not yet present in the supplied road data. The test is expected to run when verified `Distance`, `Capacity`, and `Cost` values are added. These values should not be guessed.

## Data and interpretation

The CSV files in this repository are sample project datasets. Results depend on the supplied data and the assumptions implemented in the algorithms. In particular:

- Traffic and travel-time outputs are based on the project's dataset and model assumptions.
- Transit capacity and fleet-allocation outputs are scenario estimates, not published passenger counts or official service levels.
- Missing road links or incomplete input data can limit network-optimization results.

## Contributing

Contributions and issue reports are welcome. Before submitting a change:

1. Explain the problem or proposed improvement.
2. Keep algorithm changes covered by focused tests.
3. Run the test suite.
4. Do not add credentials, private datasets, or generated environment files.

## License

No license file is currently included in this repository. Unless a license is added, the source code remains subject to the default copyright terms; reuse and redistribution permissions have not been granted.
