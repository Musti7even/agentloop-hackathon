'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  HomeIcon, 
  ChartBarIcon, 
  CogIcon 
} from '@heroicons/react/24/outline'

export default function Navigation() {
  const pathname = usePathname()

  const navigation = [
    {
      name: 'Control Dashboard',
      href: '/control',
      icon: CogIcon,
      description: 'Create and manage projects'
    },
    {
      name: 'Results Dashboard', 
      href: '/',
      icon: ChartBarIcon,
      description: 'View experiment results'
    }
  ]

  return (
    <nav className="bg-gray-900 border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">ML</span>
              </div>
              <span className="text-white font-semibold">Control System</span>
            </div>

            <div className="flex space-x-1">
              {navigation.map((item) => {
                const isActive = pathname === item.href
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-800'
                    }`}
                  >
                    <item.icon className="h-4 w-4" />
                    <span>{item.name}</span>
                  </Link>
                )
              })}
            </div>
          </div>

          <div className="text-xs text-gray-400">
            AI-Powered Outreach Platform
          </div>
        </div>
      </div>
    </nav>
  )
} 