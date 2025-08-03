import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import { projectManager } from '@/lib/project-manager'

export async function POST(request: NextRequest) {
  try {
    const { projectId, personaFile, messagesPerPersona } = await request.json()

    if (!projectId || !personaFile) {
      return NextResponse.json({ error: 'Missing required parameters' }, { status: 400 })
    }

    // Load project metadata to ensure it exists
    const project = await projectManager.loadProjectMetadata(projectId)
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 })
    }

    // Update project status
    await projectManager.updateProjectStatus(projectId, 'training')

    // Ensure project directories exist
    const projectPersonasDir = projectManager.getProjectPersonasDir(projectId)
    const projectResultsDir = projectManager.getProjectResultsDir(projectId)
    
    if (!require('fs').existsSync(projectPersonasDir)) {
      require('fs').mkdirSync(projectPersonasDir, { recursive: true })
    }
    if (!require('fs').existsSync(projectResultsDir)) {
      require('fs').mkdirSync(projectResultsDir, { recursive: true })
    }

    // Return streaming response to show real-time output
    const stream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder()

        // Send initial message
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ 
          type: 'info',
          message: `Starting performance evaluation for "${project.name}"...`,
          timestamp: new Date().toISOString()
        })}\n\n`))

        // Create a Python script that safely handles the parameters and project structure
        const pythonCode = `
import sys
import os
import json
from pathlib import Path
sys.path.append('${path.join(process.cwd(), '..', 'data_gen')}')

# Parameters
PROJECT_ID = """${projectId}"""
PERSONA_FILE = """${personaFile}"""
MESSAGES_PER_PERSONA = ${messagesPerPersona || 1}
PROJECT_PERSONAS_DIR = """${projectPersonasDir.replace(/\\/g, '/')}"""
PROJECT_RESULTS_DIR = """${projectResultsDir.replace(/\\/g, '/')}"""

try:
    # Import the module by filename
    import importlib.util
    spec = importlib.util.spec_from_file_location("end_to_end_processor", "${path.join(process.cwd(), '..', 'data_gen', 'end_to_end_processor.py')}")
    end_to_end_processor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(end_to_end_processor)
    
    print("Initializing EndToEndProcessor...")
    processor = end_to_end_processor.EndToEndProcessor(max_workers=4)
    
    # Override the data directories to point to project folders
    processor.personas_dir = Path(PROJECT_PERSONAS_DIR)
    processor.results_dir = Path(PROJECT_RESULTS_DIR)
    
    # Ensure results directory exists
    processor.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Personas directory: {processor.personas_dir}")
    print(f"Results directory: {processor.results_dir}")
    print(f"Processing persona file: {PERSONA_FILE}")
    print(f"Messages per persona: {MESSAGES_PER_PERSONA}")
    
    # Process all personas
    results = processor.process_all_personas(
        PERSONA_FILE,
        MESSAGES_PER_PERSONA
    )
    
    # Save results
    output_file = processor.save_results(results, PERSONA_FILE)
    
    # Print summary
    metadata = results["metadata"]
    print("\\n" + "="*60)
    print("PERFORMANCE EVALUATION COMPLETE")
    print("="*60)
    print(f"Persona file: {metadata['persona_file']}")
    print(f"Messages per persona: {metadata['messages_per_persona']}")
    print(f"Total personas: {metadata['total_personas']}")
    print(f"Successfully processed: {metadata['successful_personas']}")
    print(f"Total messages generated: {metadata['total_messages']}")
    print(f"Results saved to: {output_file}")
    
    # Calculate accuracy
    total_positive = sum(
        1 for persona_result in results["results"]
        for message_result in persona_result["messages"]
        if message_result["decision"]["decision"]
    )
    total_messages = metadata["total_messages"]
    if total_messages > 0:
        accuracy = (total_positive / total_messages) * 100
        print(f"Accuracy: {total_positive}/{total_messages} ({accuracy:.1f}%)")
    
    print("="*60)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
`

        // Create a temporary Python file to avoid string escaping issues
        const tempScriptPath = path.join(process.cwd(), '..', 'temp_evaluate_performance.py')
        require('fs').writeFileSync(tempScriptPath, pythonCode)

        // Spawn Python process
        const pythonProcess = spawn('python3', [tempScriptPath], {
          cwd: path.join(process.cwd(), '..'),
          stdio: ['pipe', 'pipe', 'pipe']
        })

        // Handle stdout
        pythonProcess.stdout.on('data', (data) => {
          const output = data.toString()
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ 
            type: 'stdout',
            message: output.trim(),
            timestamp: new Date().toISOString()
          })}\n\n`))
        })

        // Handle stderr
        pythonProcess.stderr.on('data', (data) => {
          const output = data.toString()
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ 
            type: 'stderr',
            message: output.trim(),
            timestamp: new Date().toISOString()
          })}\n\n`))
        })

        // Handle process completion
        pythonProcess.on('close', async (code) => {
          // Clean up temporary file
          try {
            require('fs').unlinkSync(tempScriptPath)
          } catch (error) {
            console.warn('Could not clean up temp file:', error)
          }

          if (code === 0) {
            // Update project status on success
            try {
              await projectManager.updateProjectStatus(projectId, 'ready', {
                lastEvaluationDate: new Date().toISOString()
              })
            } catch (error) {
              console.error('Error updating project status:', error)
            }
          }
          
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ 
            type: code === 0 ? 'success' : 'error',
            message: `Performance evaluation completed with exit code: ${code}`,
            timestamp: new Date().toISOString(),
            finished: true
          })}\n\n`))
          controller.close()
        })

        // Handle process errors
        pythonProcess.on('error', (error) => {
          // Clean up temporary file
          try {
            require('fs').unlinkSync(tempScriptPath)
          } catch (cleanupError) {
            console.warn('Could not clean up temp file:', cleanupError)
          }

          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ 
            type: 'error',
            message: `Process error: ${error.message}`,
            timestamp: new Date().toISOString(),
            finished: true
          })}\n\n`))
          controller.close()
        })
      }
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      }
    })

  } catch (error) {
    console.error('Error in evaluate-performance API:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
} 