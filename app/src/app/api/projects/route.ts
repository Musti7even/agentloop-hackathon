import { NextRequest, NextResponse } from 'next/server'
import { projectManager, ProjectMetadata } from '@/lib/project-manager'

// GET /api/projects - List all projects
export async function GET() {
  try {
    // Attempt migration on first load
    await projectManager.migrateExistingData()
    
    const projects = await projectManager.listProjects()
    return NextResponse.json(projects)
  } catch (error) {
    console.error('Error listing projects:', error)
    return NextResponse.json({ error: 'Failed to list projects' }, { status: 500 })
  }
}

// POST /api/projects - Create new project
export async function POST(request: NextRequest) {
  try {
    const { name, description, domain, personaCount } = await request.json()
    
    if (!name || !domain) {
      return NextResponse.json({ 
        error: 'name and domain are required' 
      }, { status: 400 })
    }

    const project = await projectManager.createProject({
      name,
      description: description || '',
      domain,
      personaCount: personaCount || 20
    })

    return NextResponse.json(project)
  } catch (error) {
    console.error('Error creating project:', error)
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Failed to create project' 
    }, { status: 500 })
  }
}

// PUT /api/projects/[id] - Update project
export async function PUT(request: NextRequest) {
  try {
    const url = new URL(request.url)
    const projectId = url.pathname.split('/').pop()
    
    if (!projectId) {
      return NextResponse.json({ error: 'Project ID is required' }, { status: 400 })
    }

    const updates = await request.json()
    
    const currentMetadata = await projectManager.loadProjectMetadata(projectId)
    if (!currentMetadata) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 })
    }

    // Update metadata
    Object.assign(currentMetadata, updates)
    await projectManager.saveProjectMetadata(currentMetadata)

    return NextResponse.json(currentMetadata)
  } catch (error) {
    console.error('Error updating project:', error)
    return NextResponse.json({ error: 'Failed to update project' }, { status: 500 })
  }
} 