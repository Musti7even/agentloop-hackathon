'use client'

import { useState, useRef, useEffect } from 'react'
import { 
  UserGroupIcon, 
  SparklesIcon, 
  ChartBarIcon, 
  CogIcon,
  PlusIcon,
  PlayIcon,
  DocumentTextIcon,
  BeakerIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import Terminal from '@/components/Terminal'
import { ProjectMetadata } from '@/lib/project-manager'

type Project = ProjectMetadata

interface PersonaGenerationForm {
  projectName: string
  description: string
  domainContext: string
  personaCount: number
}

export default function ControlDashboard() {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [showNewProjectForm, setShowNewProjectForm] = useState(false)
  const [terminalOpen, setTerminalOpen] = useState(false)
  const [terminalType, setTerminalType] = useState<'personas' | 'improvement'>('personas')
  const [terminalData, setTerminalData] = useState<any>({})
  
  const [newProject, setNewProject] = useState<PersonaGenerationForm>({
    projectName: '',
    description: '',
    domainContext: '',
    personaCount: 20
  })

  const formRef = useRef<HTMLDivElement>(null)

  // Load projects on component mount
  useEffect(() => {
    loadProjects()
  }, [])

  const loadProjects = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/projects')
      if (response.ok) {
        const projectsData = await response.json()
        setProjects(projectsData)
      }
    } catch (error) {
      console.error('Error loading projects:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCreateProject = async () => {
    if (!newProject.projectName || !newProject.domainContext) return

    try {
      const response = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newProject.projectName,
          description: newProject.description,
          domain: newProject.domainContext,
          personaCount: newProject.personaCount
        })
      })

      if (response.ok) {
        const project = await response.json()
        setProjects(prev => [project, ...prev])
        setShowNewProjectForm(false)
        setNewProject({
          projectName: '',
          description: '',
          domainContext: '',
          personaCount: 20
        })
      }
    } catch (error) {
      console.error('Error creating project:', error)
    }
  }

  const handleGeneratePersonas = (project: Project) => {
    // Update project status locally (will be updated by API)
    setProjects(prev => prev.map(p => 
      p.id === project.id 
        ? { ...p, status: 'generating_personas' as const }
        : p
    ))

    // Open terminal for persona generation
    setTerminalType('personas')
    setTerminalData({
      projectId: project.id,
      domainContext: project.domain,
      count: project.config.personaGeneration.count,
      filename: project.name.toLowerCase().replace(/\s+/g, '_')
    })
    setTerminalOpen(true)
  }

  const handleImprovePrompts = (project: Project) => {
    setTerminalType('improvement')
    setTerminalData({
      projectId: project.id,
      fileName: `${project.name.toLowerCase().replace(/\s+/g, '_')}_results.json`
    })
    setTerminalOpen(true)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'draft': return 'bg-gray-500/10 text-gray-400 border-gray-500/20'
      case 'generating_personas': return 'bg-blue-500/10 text-blue-400 border-blue-500/20'
      case 'personas_ready': return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
      case 'training': return 'bg-purple-500/10 text-purple-400 border-purple-500/20'
      case 'ready': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'draft': return <DocumentTextIcon className="h-4 w-4" />
      case 'generating_personas': return <UserGroupIcon className="h-4 w-4 animate-pulse" />
      case 'personas_ready': return <UserGroupIcon className="h-4 w-4" />
      case 'training': return <BeakerIcon className="h-4 w-4 animate-pulse" />
      case 'ready': return <CheckCircleIcon className="h-4 w-4" />
      default: return <ExclamationTriangleIcon className="h-4 w-4" />
    }
  }

  const getNextAction = (project: Project) => {
    switch (project.status) {
      case 'draft':
        return {
          label: 'Generate Personas',
          action: () => handleGeneratePersonas(project),
          icon: <UserGroupIcon className="h-4 w-4" />,
          color: 'bg-blue-600 hover:bg-blue-700'
        }
      case 'personas_ready':
        return {
          label: 'Start Training',
          action: () => handleImprovePrompts(project),
          icon: <BeakerIcon className="h-4 w-4" />,
          color: 'bg-purple-600 hover:bg-purple-700'
        }
      case 'ready':
        return {
          label: 'Improve Model',
          action: () => handleImprovePrompts(project),
          icon: <SparklesIcon className="h-4 w-4" />,
          color: 'bg-emerald-600 hover:bg-emerald-700'
        }
      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white mb-1">
                ML Control Dashboard
              </h1>
              <p className="text-gray-400">
                Create and manage your AI-powered outreach projects
              </p>
            </div>
            <button
              onClick={() => setShowNewProjectForm(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              <PlusIcon className="h-4 w-4" />
              <span>New Project</span>
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <DocumentTextIcon className="h-5 w-5 text-blue-400" />
              </div>
            </div>
            <div className="text-2xl font-bold text-white mb-1">{projects.length}</div>
            <div className="text-sm text-gray-400">Total Projects</div>
          </div>

          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-emerald-500/10 rounded-lg">
                <CheckCircleIcon className="h-5 w-5 text-emerald-400" />
              </div>
            </div>
            <div className="text-2xl font-bold text-white mb-1">
              {projects.filter(p => p.status === 'ready').length}
            </div>
            <div className="text-sm text-gray-400">Ready Models</div>
          </div>

          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-purple-500/10 rounded-lg">
                <UserGroupIcon className="h-5 w-5 text-purple-400" />
              </div>
            </div>
            <div className="text-2xl font-bold text-white mb-1">
              {projects.reduce((sum, p) => sum + (p.personaCount || 0), 0)}
            </div>
            <div className="text-sm text-gray-400">Total Personas</div>
          </div>

          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-yellow-500/10 rounded-lg">
                <ChartBarIcon className="h-5 w-5 text-yellow-400" />
              </div>
            </div>
            <div className="text-2xl font-bold text-white mb-1">
              {projects.filter(p => p.accuracy).length > 0 
                ? Math.round(projects.filter(p => p.accuracy).reduce((sum, p) => sum + (p.accuracy || 0), 0) / projects.filter(p => p.accuracy).length * 10) / 10
                : 0}%
            </div>
            <div className="text-sm text-gray-400">Avg Accuracy</div>
          </div>
        </div>

        {/* Projects List */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl backdrop-blur">
          <div className="px-6 py-4 border-b border-gray-800">
            <h2 className="text-lg font-semibold text-white">Projects</h2>
            <p className="text-sm text-gray-400 mt-1">
              Manage your AI outreach projects from ideation to deployment
            </p>
          </div>

          <div className="p-6">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                <span className="ml-3 text-gray-400">Loading projects...</span>
              </div>
            ) : projects.length === 0 ? (
              <div className="text-center py-12">
                <DocumentTextIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-300 mb-2">No projects yet</h3>
                <p className="text-gray-500 mb-4">Create your first AI outreach project to get started</p>
                <button
                  onClick={() => setShowNewProjectForm(true)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
                >
                  Create Project
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                {projects.map((project) => {
                const nextAction = getNextAction(project)
                
                return (
                  <div
                    key={project.id}
                    className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 hover:bg-gray-800/70 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-3">
                          <h3 className="text-lg font-semibold text-white">
                            {project.name}
                          </h3>
                          <span className={`inline-flex items-center space-x-1 px-2 py-1 text-xs font-medium rounded-full border ${getStatusColor(project.status)}`}>
                            {getStatusIcon(project.status)}
                            <span className="capitalize">{project.status.replace('_', ' ')}</span>
                          </span>
                        </div>
                        
                        <p className="text-gray-300 mb-2">{project.description}</p>
                        <p className="text-sm text-gray-400 mb-4">
                          <span className="font-medium">Domain:</span> {project.domain}
                        </p>

                        <div className="flex items-center space-x-6 text-sm text-gray-400">
                          <span>Created {new Date(project.createdAt).toLocaleDateString()}</span>
                          {project.personaCount && (
                            <span>{project.personaCount} personas</span>
                          )}
                          {project.accuracy && (
                            <span className="text-emerald-400 font-medium">
                              {project.accuracy}% accuracy
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center space-x-3 ml-6">
                        {nextAction && (
                          <button
                            onClick={nextAction.action}
                            className={`flex items-center space-x-2 px-4 py-2 text-white rounded-lg font-medium transition-colors ${nextAction.color}`}
                          >
                            {nextAction.icon}
                            <span>{nextAction.label}</span>
                          </button>
                        )}
                        
                        <button className="p-2 text-gray-400 hover:text-gray-300 transition-colors">
                          <CogIcon className="h-5 w-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                )
              })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* New Project Form Modal */}
      {showNewProjectForm && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur flex items-center justify-center p-4 z-40">
          <div ref={formRef} className="bg-gray-900 border border-gray-800 rounded-xl max-w-2xl w-full">
            <div className="px-6 py-4 border-b border-gray-800">
              <h3 className="text-lg font-semibold text-white">Create New Project</h3>
              <p className="text-sm text-gray-400 mt-1">
                Set up a new AI-powered outreach project
              </p>
            </div>

            <div className="p-6 space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Project Name
                </label>
                <input
                  type="text"
                  value={newProject.projectName}
                  onChange={(e) => setNewProject(prev => ({ ...prev, projectName: e.target.value }))}
                  placeholder="e.g., WorkyAI Cold Outreach"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Description
                </label>
                <input
                  type="text"
                  value={newProject.description}
                  onChange={(e) => setNewProject(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Brief description of your outreach project"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Domain Context
                </label>
                <textarea
                  value={newProject.domainContext}
                  onChange={(e) => setNewProject(prev => ({ ...prev, domainContext: e.target.value }))}
                  placeholder="Describe your target market, industry, and use case. E.g., 'B2B SaaS companies looking for workflow automation tools because of operational overhead. From small startups to large enterprises.'"
                  rows={4}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Number of Personas
                </label>
                <input
                  type="number"
                  value={newProject.personaCount}
                  onChange={(e) => setNewProject(prev => ({ ...prev, personaCount: parseInt(e.target.value) || 20 }))}
                  min="5"
                  max="100"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-400 mt-1">
                  Recommended: 20-50 personas for balanced training
                </p>
              </div>
            </div>

            <div className="px-6 py-4 border-t border-gray-800 flex justify-end space-x-3">
              <button
                onClick={() => setShowNewProjectForm(false)}
                className="px-4 py-2 text-gray-400 hover:text-gray-300 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateProject}
                disabled={!newProject.projectName || !newProject.domainContext}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
              >
                Create Project
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Terminal Component */}
      {terminalOpen && terminalType === 'personas' && (
        <PersonaTerminal
          isOpen={terminalOpen}
          onClose={() => {
            setTerminalOpen(false)
            // Refresh projects to get updated status
            loadProjects()
          }}
          data={terminalData}
        />
      )}

      {terminalType === 'improvement' && (
        <Terminal
          isOpen={terminalOpen}
          onClose={() => setTerminalOpen(false)}
          fileName={terminalData.fileName}
        />
      )}
    </div>
  )
}

// Persona Generation Terminal Component
function PersonaTerminal({ isOpen, onClose, data }: { isOpen: boolean, onClose: () => void, data: any }) {
  const [messages, setMessages] = useState<any[]>([])
  const [isRunning, setIsRunning] = useState(false)

  const startGeneration = async () => {
    if (isRunning) return

    setIsRunning(true)
    setMessages([])

    try {
      const response = await fetch('/api/generate-personas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      })

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              setMessages(prev => [...prev, data])
              
              if (data.finished) {
                setIsRunning(false)
              }
            } catch (e) {
              // Ignore invalid JSON
            }
          }
        }
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        message: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toISOString()
      }])
      setIsRunning(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur flex items-center justify-center p-4 z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-6xl h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <UserGroupIcon className="h-5 w-5 text-gray-400" />
            <h2 className="text-lg font-semibold text-white">Persona Generation</h2>
          </div>
          <div className="flex items-center space-x-3">
            {!isRunning && (
              <button
                onClick={startGeneration}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
              >
                Start Generation
              </button>
            )}
            <button onClick={onClose} className="text-gray-400 hover:text-gray-300">
              ✕
            </button>
          </div>
        </div>

        {/* Terminal Content */}
        <div className="flex-1 bg-black rounded-b-xl overflow-hidden">
          <div className="p-4 h-full overflow-y-auto font-mono text-sm">
            {messages.length === 0 && !isRunning && (
              <div className="text-gray-500 text-center py-8">
                Click "Start Generation" to create personas for your project...
              </div>
            )}
            
            {messages.map((msg, index) => (
              <div key={index} className="mb-1 flex">
                <span className="text-gray-500 text-xs w-20">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </span>
                <span className={`flex-1 ${
                  msg.type === 'info' ? 'text-blue-400' :
                  msg.type === 'stdout' ? 'text-green-400' :
                  msg.type === 'stderr' ? 'text-yellow-400' :
                  msg.type === 'error' ? 'text-red-400' :
                  msg.type === 'success' ? 'text-emerald-400' : 'text-gray-300'
                }`}>
                  {msg.message}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
} 