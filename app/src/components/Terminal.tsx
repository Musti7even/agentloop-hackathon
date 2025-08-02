'use client'

import { useState, useEffect, useRef } from 'react'
import { XMarkIcon, CommandLineIcon } from '@heroicons/react/24/outline'

interface TerminalMessage {
  type: 'info' | 'stdout' | 'stderr' | 'error' | 'success'
  message: string
  timestamp: string
  finished?: boolean
}

interface TerminalProps {
  isOpen: boolean
  onClose: () => void
  fileName: string
}

export default function Terminal({ isOpen, onClose, fileName }: TerminalProps) {
  const [messages, setMessages] = useState<TerminalMessage[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const startImprovement = async () => {
    if (isRunning) return

    setIsRunning(true)
    setMessages([])

    try {
      const response = await fetch('/api/improve-prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fileName }),
      })

      if (!response.ok) {
        throw new Error('Failed to start improvement process')
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }

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

  const getMessageColor = (type: string) => {
    switch (type) {
      case 'info': return 'text-blue-400'
      case 'stdout': return 'text-green-400'
      case 'stderr': return 'text-yellow-400'
      case 'error': return 'text-red-400'
      case 'success': return 'text-emerald-400'
      default: return 'text-gray-300'
    }
  }

  const getTypePrefix = (type: string) => {
    switch (type) {
      case 'info': return '[INFO]'
      case 'stdout': return '[OUT]'
      case 'stderr': return '[WARN]'
      case 'error': return '[ERROR]'
      case 'success': return '[SUCCESS]'
      default: return '[LOG]'
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur flex items-center justify-center p-4 z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-6xl h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <CommandLineIcon className="h-5 w-5 text-gray-400" />
            <h2 className="text-lg font-semibold text-white">
              Prompt Improvement Terminal
            </h2>
            <span className="text-sm text-gray-400">({fileName})</span>
          </div>
          <div className="flex items-center space-x-3">
            {!isRunning && (
              <button
                onClick={startImprovement}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors"
              >
                Start Improvement
              </button>
            )}
            {isRunning && (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-500"></div>
                <span className="text-green-400 text-sm">Running...</span>
              </div>
            )}
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-300 transition-colors"
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Terminal Content */}
        <div className="flex-1 bg-black rounded-b-xl overflow-hidden">
          <div className="p-4 h-full overflow-y-auto font-mono text-sm">
            {messages.length === 0 && !isRunning && (
              <div className="text-gray-500 text-center py-8">
                Click "Start Improvement" to begin the prompt improvement process...
              </div>
            )}
            
            {messages.map((msg, index) => (
              <div key={index} className="mb-1 flex">
                <span className="text-gray-500 text-xs w-20 flex-shrink-0">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </span>
                <span className={`text-xs w-16 flex-shrink-0 ${getMessageColor(msg.type)}`}>
                  {getTypePrefix(msg.type)}
                </span>
                <span className={`flex-1 ${getMessageColor(msg.type)}`}>
                  {msg.message}
                </span>
              </div>
            ))}
            
            {isRunning && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="animate-pulse">▋</div>
                <span>Process running...</span>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-gray-700 bg-gray-800/50">
          <div className="flex items-center justify-between text-xs text-gray-400">
            <span>Prompt Improvement Service</span>
            <span>{messages.length} messages</span>
          </div>
        </div>
      </div>
    </div>
  )
} 