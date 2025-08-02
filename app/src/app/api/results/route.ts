import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

interface DecisionResult {
  decision: boolean;
  reasoning: string;
}

interface MessageResult {
  message: string;
  decision: DecisionResult;
}

interface PersonaResult {
  persona: string;
  messages: MessageResult[];
}

interface ResultsFile {
  metadata: {
    messages_per_persona: number;
    total_personas: number;
    total_messages: number;
    processing_date: string;
    system_prompt: string;
    improvement_run?: boolean;
    successful_personas: number;
    persona_file?: string;
  };
  results: PersonaResult[];
}

interface AnalyzedResult {
  fileName: string;
  runName: string;
  accuracy: number;
  trueDecisions: number;
  falseDecisions: number;
  totalMessages: number;
  totalPersonas: number;
  isImprovementRun: boolean;
  processingDate: string;
  systemPrompt: string;
}

function analyzeResultsFile(filePath: string, fileName: string): AnalyzedResult | null {
  try {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const data: ResultsFile = JSON.parse(fileContent);
    
    // Calculate actual metrics from results data, not metadata
    let trueDecisions = 0;
    let falseDecisions = 0;
    let totalMessages = 0;
    
    for (const personaResult of data.results) {
      for (const messageResult of personaResult.messages) {
        totalMessages++;
        if (messageResult.decision.decision) {
          trueDecisions++;
        } else {
          falseDecisions++;
        }
      }
    }
    
    const accuracy = totalMessages > 0 ? (trueDecisions / totalMessages) * 100 : 0;
    
    // Generate a clean run name
    let runName = fileName.replace('.json', '');
    if (runName.includes('improvement_run')) {
      const versionMatch = runName.match(/improvement_run_v(\d+)/);
      runName = versionMatch ? `Improvement Run v${versionMatch[1]}` : 'Improvement Run';
    } else if (runName.includes('_results_')) {
      const timestampMatch = runName.match(/_(\d{8}_\d{6})$/);
      runName = timestampMatch ? `Original Run (${timestampMatch[1]})` : 'Original Run';
    } else {
      runName = 'Original Run';
    }
    
    return {
      fileName,
      runName,
      accuracy: Math.round(accuracy * 10) / 10, // Round to 1 decimal
      trueDecisions,
      falseDecisions,
      totalMessages,
      totalPersonas: data.results.length,
      isImprovementRun: data.metadata.improvement_run || false,
      processingDate: data.metadata.processing_date,
      systemPrompt: data.metadata.system_prompt || ''
    };
  } catch (error) {
    console.error(`Error analyzing file ${fileName}:`, error);
    return null;
  }
}

export async function GET() {
  try {
    // Read from the actual data/results directory
    const resultsDir = path.join(process.cwd(), '..', 'data', 'results');
    
    if (!fs.existsSync(resultsDir)) {
      return NextResponse.json({ error: 'Results directory not found' }, { status: 404 });
    }
    
    const files = fs.readdirSync(resultsDir);
    const jsonFiles = files.filter(file => file.endsWith('.json'));
    
    const analyzedResults: AnalyzedResult[] = [];
    
    for (const file of jsonFiles) {
      const filePath = path.join(resultsDir, file);
      const analyzed = analyzeResultsFile(filePath, file);
      if (analyzed) {
        analyzedResults.push(analyzed);
      }
    }
    
    // Sort by processing date (newest first)
    analyzedResults.sort((a, b) => new Date(b.processingDate).getTime() - new Date(a.processingDate).getTime());
    
    return NextResponse.json(analyzedResults);
  } catch (error) {
    console.error('Error reading results directory:', error);
    return NextResponse.json({ error: 'Failed to read results' }, { status: 500 });
  }
} 