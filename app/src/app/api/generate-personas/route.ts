import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import { projectManager } from '@/lib/project-manager'

export async function POST(request: NextRequest) {
  try {
    const { projectId, domainContext, count, filename } = await request.json()
    
    if (!projectId || !domainContext || !filename) {
      return NextResponse.json({ 
        error: 'projectId, domainContext and filename are required' 
      }, { status: 400 })
    }

    // Load project metadata
    const project = await projectManager.loadProjectMetadata(projectId)
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 })
    }

    // Update project status
    await projectManager.updateProjectStatus(projectId, 'generating_personas')

    // Ensure project directories exist
    const projectPersonasDir = projectManager.getProjectPersonasDir(projectId)
    if (!require('fs').existsSync(projectPersonasDir)) {
      require('fs').mkdirSync(projectPersonasDir, { recursive: true })
    }

    // Return streaming response to show real-time output
    const stream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder()
        
        // Send initial message
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ 
          type: 'info', 
          message: `Starting persona generation for "${filename}"...`,
          timestamp: new Date().toISOString()
        })}\n\n`))

        // Create a Python script that safely handles the parameters
        const pythonCode = `
import sys
import os
import json
sys.path.append('${path.join(process.cwd(), '..', 'data_gen')}')

# Parameters passed safely through environment or arguments
PROJECT_ID = """${projectId}"""
DOMAIN_CONTEXT = """${domainContext.replace(/"""/g, '\\"""')}"""
COUNT = ${count || 20}
FILENAME = """${filename}"""
DATA_DIR = """${projectManager.getProjectPersonasDir(projectId).replace(/\\/g, '/')}"""

try:
    # Import the module by filename
    import importlib.util
    spec = importlib.util.spec_from_file_location("datagen_service", "${path.join(process.cwd(), '..', 'data_gen', 'datagen-service.py')}")
    datagen_service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datagen_service)
    
    print("Initializing DataGenerationService...")
    service = datagen_service.DataGenerationService()
    
    # Override the data directory to point to project folder
    service.data_dir = DATA_DIR
    os.makedirs(service.data_dir, exist_ok=True)
    print(f"Data directory set to: {service.data_dir}")
    
    print(f"Generating {COUNT} personas for project: {PROJECT_ID}")
    print(f"Domain: {DOMAIN_CONTEXT}")
    
    personas = service.generate_personas_sync(
        domain_context=DOMAIN_CONTEXT,
        count=COUNT,
        filename=FILENAME
    )
    
    print(f"Successfully generated {len(personas)} personas!")
    print(f"Personas saved to project directory: {service.data_dir}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
`

        // Create a temporary Python file to avoid string escaping issues
        const tempScriptPath = path.join(process.cwd(), '..', 'temp_persona_gen.py')
        require('fs').writeFileSync(tempScriptPath, pythonCode)

        // Spawn Python process
        const pythonProcess = spawn('python3', [tempScriptPath], {
          cwd: path.join(process.cwd(), '..'),
          stdio: ['pipe', 'pipe', 'pipe']
        })

        // Handle stdout
        pythonProcess.stdout?.on('data', (data) => {
          const message = data.toString()
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ 
            type: 'stdout', 
            message: message.trim(),
            timestamp: new Date().toISOString()
          })}\n\n`))
        })

        // Handle stderr
        pythonProcess.stderr?.on('data', (data) => {
          const message = data.toString()
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ 
            type: 'stderr', 
            message: message.trim(),
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
              await projectManager.updateProjectStatus(projectId, 'personas_ready', {
                personaCount: count || 20
              })
            } catch (error) {
              console.error('Error updating project status:', error)
            }
          }
          
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ 
            type: code === 0 ? 'success' : 'error',
            message: `Persona generation completed with exit code: ${code}`,
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
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    })

  } catch (error) {
    console.error('Error starting persona generation:', error)
    return NextResponse.json(
      { error: 'Failed to start persona generation process' }, 
      { status: 500 }
    )
  }
} 