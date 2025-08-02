import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  try {
    const { domainContext, count, filename } = await request.json()
    
    if (!domainContext || !filename) {
      return NextResponse.json({ 
        error: 'domainContext and filename are required' 
      }, { status: 400 })
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

        // Create a Python script call that imports and uses the DataGenerationService
        const pythonCode = `
import sys
import os
sys.path.append('${path.join(process.cwd(), '..', 'data_gen')}')

try:
    # Import the module by filename
    import importlib.util
    spec = importlib.util.spec_from_file_location("datagen_service", "${path.join(process.cwd(), '..', 'data_gen', 'datagen-service.py')}")
    datagen_service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datagen_service)
    
    print("Initializing DataGenerationService...")
    service = datagen_service.DataGenerationService()
    
    print(f"Generating ${count || 20} personas for domain: ${domainContext}")
    personas = service.generate_personas_sync(
        domain_context="${domainContext}",
        count=${count || 20},
        filename="${filename}"
    )
    
    print(f"Successfully generated {len(personas)} personas!")
    print("Personas saved to data/personas directory")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
`

        // Spawn Python process
        const pythonProcess = spawn('python3', ['-c', pythonCode], {
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
        pythonProcess.on('close', (code) => {
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