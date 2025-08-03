import { NextRequest, NextResponse } from 'next/server'
import { projectManager } from '@/lib/project-manager'
import fs from 'fs'
import path from 'path'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const projectId = params.id
    
    if (!projectId) {
      return NextResponse.json({ error: 'Project ID is required' }, { status: 400 })
    }

    // Check if project exists
    const project = await projectManager.loadProjectMetadata(projectId)
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 })
    }

    // Get results directory for the project
    const resultsDir = projectManager.getProjectResultsDir(projectId)
    
    if (!fs.existsSync(resultsDir)) {
      return NextResponse.json({ results: [], files: [] })
    }

    // List all result files
    const files = fs.readdirSync(resultsDir).filter(file => file.endsWith('.json'))
    const results: any[] = []

    // Load metadata from results files
    for (const file of files) {
      try {
        const filePath = path.join(resultsDir, file)
        const content = fs.readFileSync(filePath, 'utf-8')
        const resultData = JSON.parse(content)
        
        if (resultData.metadata) {
          // Calculate accuracy
          let accuracy = 0
          if (resultData.results && resultData.results.length > 0) {
            const totalPositive = resultData.results.reduce((total: number, persona: any) => {
              return total + persona.messages.reduce((msgTotal: number, msg: any) => {
                return msgTotal + (msg.decision.decision ? 1 : 0)
              }, 0)
            }, 0)
            const totalMessages = resultData.metadata.total_messages || 0
            accuracy = totalMessages > 0 ? (totalPositive / totalMessages) * 100 : 0
          }

          results.push({
            file,
            metadata: resultData.metadata,
            accuracy: Math.round(accuracy * 10) / 10,
            createdAt: resultData.metadata.processing_date || fs.statSync(filePath).birthtime.toISOString()
          })
        }
      } catch (error) {
        console.error(`Error reading result file ${file}:`, error)
      }
    }

    // Sort by creation date (newest first)
    results.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())

    return NextResponse.json({
      results,
      files,
      totalFiles: files.length,
      projectId
    })

  } catch (error) {
    console.error('Error listing project results:', error)
    return NextResponse.json({ error: 'Failed to list results' }, { status: 500 })
  }
} 