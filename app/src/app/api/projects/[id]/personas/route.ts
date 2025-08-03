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

    // Get personas directory for the project
    const personasDir = projectManager.getProjectPersonasDir(projectId)
    
    if (!fs.existsSync(personasDir)) {
      return NextResponse.json({ personas: [], files: [] })
    }

    // List all persona files
    const files = fs.readdirSync(personasDir).filter(file => file.endsWith('.json'))
    const personas: any[] = []

    // Load personas from files
    for (const file of files) {
      try {
        const filePath = path.join(personasDir, file)
        const content = fs.readFileSync(filePath, 'utf-8')
        const personaData = JSON.parse(content)
        
        if (Array.isArray(personaData)) {
          personas.push({
            file,
            count: personaData.length,
            personas: personaData.slice(0, 3) // First 3 for preview
          })
        }
      } catch (error) {
        console.error(`Error reading persona file ${file}:`, error)
      }
    }

    return NextResponse.json({
      personas,
      files,
      totalFiles: files.length,
      projectId
    })

  } catch (error) {
    console.error('Error listing project personas:', error)
    return NextResponse.json({ error: 'Failed to list personas' }, { status: 500 })
  }
} 