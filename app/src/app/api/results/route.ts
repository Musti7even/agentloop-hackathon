import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'
import { projectManager } from '@/lib/project-manager'

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
  projectId?: string;
  projectName?: string;
}

function analyzeResultsFile(filePath: string, fileName: string, projectId?: string, projectName?: string): AnalyzedResult | null {
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
      systemPrompt: data.metadata.system_prompt || '',
      projectId,
      projectName
    };
  } catch (error) {
    console.error(`Error analyzing file ${fileName}:`, error);
    return null;
  }
}

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const projectId = url.searchParams.get('projectId');
    
    const analyzedResults: AnalyzedResult[] = [];

    if (projectId) {
      // Get results for a specific project
      const project = await projectManager.loadProjectMetadata(projectId);
      if (!project) {
        return NextResponse.json({ error: 'Project not found' }, { status: 404 });
      }

      const projectResultsDir = projectManager.getProjectResultsDir(projectId);
      if (fs.existsSync(projectResultsDir)) {
        const files = fs.readdirSync(projectResultsDir).filter(file => file.endsWith('.json'));
        
        for (const file of files) {
          const filePath = path.join(projectResultsDir, file);
          const analyzedResult = analyzeResultsFile(filePath, file, projectId, project.name);
          if (analyzedResult) {
            analyzedResults.push(analyzedResult);
          }
        }
      }
    } else {
      // Get results from all projects + legacy flat structure
      
      // 1. Get all projects and their results
      const projects = await projectManager.listProjects();
      for (const project of projects) {
        const projectResultsDir = projectManager.getProjectResultsDir(project.id);
        if (fs.existsSync(projectResultsDir)) {
          const files = fs.readdirSync(projectResultsDir).filter(file => file.endsWith('.json'));
          
          for (const file of files) {
            const filePath = path.join(projectResultsDir, file);
            const analyzedResult = analyzeResultsFile(filePath, file, project.id, project.name);
            if (analyzedResult) {
              analyzedResults.push(analyzedResult);
            }
          }
        }
      }

      // 2. Also include legacy flat results (if any)
      const legacyResultsDir = path.join(process.cwd(), '..', 'data', 'results');
      if (fs.existsSync(legacyResultsDir)) {
        const files = fs.readdirSync(legacyResultsDir).filter(file => file.endsWith('.json'));
        
        for (const file of files) {
          const filePath = path.join(legacyResultsDir, file);
          const analyzedResult = analyzeResultsFile(filePath, file, 'legacy-project', 'Legacy Project');
          if (analyzedResult) {
            analyzedResults.push(analyzedResult);
          }
        }
      }
    }

    // Sort by processing date (newest first)
    analyzedResults.sort((a, b) => new Date(b.processingDate).getTime() - new Date(a.processingDate).getTime());

    return NextResponse.json(analyzedResults);
  } catch (error) {
    console.error('Error fetching results:', error);
    return NextResponse.json({ error: 'Failed to fetch results' }, { status: 500 });
  }
} 