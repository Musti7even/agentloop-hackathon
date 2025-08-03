# Prompt Improvement Service

The `prompt_improvement_service.py` is a self-improving system that analyzes decision maker reasoning to iteratively improve cold outreach message generation prompts.

## How It Works

1. **Analyzes Results**: Takes existing results files from `/data/results/` that contain decision maker reasoning
2. **Extracts Feedback**: Collects all reasoning from positive and negative decisions
3. **Improves Prompt**: Uses an LLM to analyze patterns and improve the system prompt
4. **Re-evaluates**: Generates new messages with the improved prompt and tests them
5. **Saves Results**: Creates a versioned improvement file with metrics

## Usage

### Basic Usage
```bash
python3 data_gen/prompt_improvement_service.py --results-file b2b_saas_personas_train_results_20250802_144126.json
```

### With Custom Messages Per Persona
```bash
python3 data_gen/prompt_improvement_service.py --results-file your_results_file.json --messages-per-persona 3
```

## Parameters

- `--results-file`: Name of the results JSON file in `data/results/` directory (required)
- `--messages-per-persona`: Number of messages to generate per persona in improvement run (default: 1)

## Output

The service creates:
- A new results file: `{original_name}_improvement_run_v{version}.json`
- Comprehensive logging showing the improvement process
- Metrics comparing original vs improved performance

## Example Output

```
============================================================
IMPROVEMENT CYCLE COMPLETE
============================================================
Original response rate: 100.0% (16/16)
Improved response rate: 93.8% (15/16)
Absolute improvement: -6.2 percentage points
Relative improvement: -6.2%
Results saved to: data/results/b2b_saas_personas_train_results_improvement_run_v1.json
============================================================
```

## File Structure

```
data/
├── results/
│   ├── original_results.json          # Input: Original results with reasoning
│   └── original_improvement_run_v1.json  # Output: Improved results
├── personas/
│   └── original_personas.json         # Source personas file
└── ...
```

## Key Features

- **Intelligent Analysis**: Extracts patterns from both positive and negative decision maker feedback
- **Automated Improvement**: Uses advanced LLM to refine system prompts based on real feedback
- **Parallel Processing**: Uses ThreadPoolExecutor with 6 workers for fast evaluation (3-4x speedup)
- **Complete Re-evaluation**: Tests improved prompts on the same personas for fair comparison
- **Accurate Metadata**: Correctly tracks successful personas and response rates
- **Iterative Capability**: Can run improvements on previous improvement results
- **Versioning**: Automatically versions improvement runs to prevent overwrites
- **Comprehensive Metrics**: Provides detailed before/after analysis
- **Robust Error Handling**: Continues processing even if individual messages fail

## Requirements

- `ANTHROPIC_API_KEY` environment variable set
- Python dependencies: `anthropic`, `python-dotenv`
- Existing results file with decision maker reasoning
- Original personas file referenced in the results metadata

## Integration

This service integrates with:
- `generation_model.py`: For generating improved messages
- `decision_maker.py`: For evaluating new messages
- `end_to_end_processor.py`: Can be used on results from this processor

## Self-Improvement Loop

The service can be run iteratively:
1. Run initial evaluation → `results_v1.json`
2. Run improvement → `results_improvement_run_v1.json` 
3. Run improvement on improvement → `results_improvement_run_v2.json`
4. Continue until performance converges

This creates a self-improving system that learns from decision maker feedback to optimize messaging strategies. 