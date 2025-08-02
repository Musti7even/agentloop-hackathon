import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  try {
    const { fileName } = await request.json()
    
    if (!fileName) {
      return NextResponse.json({ error: 'fileName is required' }, { status: 400 })
    }

    // Path to the prompt improvement service script
    const scriptPath = path.join(process.cwd(), '..', 'data_gen', 'prompt_improvement_service.py')
    
    // Return streaming response to show real-time output
    const stream = new ReadableStream({
      start(controller) {
        // Spawn the Python process
        const pythonProcess = spawn('python3', [
          scriptPath,
          '--results-file',
          fileName,
          '--messages-per-persona',
          '1'
        ], {
          cwd: path.join(process.cwd(), '..'),
          stdio: ['pipe', 'pipe', 'pipe']
        })

        // Send initial message
        const encoder = new TextEncoder()
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ 
          type: 'info', 
          message: `Starting prompt improvement for ${fileName}...`,
          timestamp: new Date().toISOString()
        })}\n\n`))

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
            message: `Process completed with exit code: ${code}`,
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
    console.error('Error starting prompt improvement:', error)
    return NextResponse.json(
      { error: 'Failed to start prompt improvement process' }, 
      { status: 500 }
    )
  }
} 