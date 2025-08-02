'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'

interface AnalyzedResult {
  filename: string
  runName: string
  totalMessages: number
  trueDecisions: number
  falseDecisions: number
  accuracy: number
  processingDate: string
  isImprovementRun: boolean
  systemPromptLength: number
}

// Static data for testing - we'll replace this with dynamic loading later
const staticResults: AnalyzedResult[] = [
  {
    filename: 'b2b_saas_personas_train_results_20250802_144126.json',
    runName: 'Original Run',
    totalMessages: 16,
    trueDecisions: 16,
    falseDecisions: 0,
    accuracy: 100.0,
    processingDate: '2025-08-02T14:41:07.941932',
    isImprovementRun: false,
    systemPromptLength: 498,
  },
  {
    filename: 'b2b_saas_personas_train_results_improvement_run_v1.json',
    runName: 'Improvement Run v1',
    totalMessages: 16,
    trueDecisions: 15,
    falseDecisions: 1,
    accuracy: 93.8,
    processingDate: '2025-08-02T15:09:33.409075',
    isImprovementRun: true,
    systemPromptLength: 2062,
  },
  {
    filename: 'b2b_saas_personas_train_results_improvement_run_v2.json',
    runName: 'Improvement Run v2',
    totalMessages: 16,
    trueDecisions: 13,
    falseDecisions: 3,
    accuracy: 81.2,
    processingDate: '2025-08-02T15:20:49.857537',
    isImprovementRun: true,
    systemPromptLength: 2765,
  }
]

export function ResultsDashboard() {
  const [results, setResults] = useState<AnalyzedResult[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedResult, setSelectedResult] = useState<AnalyzedResult | null>(null)

  useEffect(() => {
    // For now, use static data
    setTimeout(() => {
      setResults(staticResults)
      setLoading(false)
    }, 1000)
  }, [])

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString()
  }

  const getAccuracyColor = (accuracy: number): string => {
    if (accuracy >= 90) return 'bg-green-100 text-green-800'
    if (accuracy >= 75) return 'bg-yellow-100 text-yellow-800'
    if (accuracy >= 60) return 'bg-orange-100 text-orange-800'
    return 'bg-red-100 text-red-800'
  }

  const getRunTypeColor = (isImprovement: boolean): string => {
    return isImprovement ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading results...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Total Runs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{results.length}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Best Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-600">
              {results.length > 0 ? Math.max(...results.map(r => r.accuracy)).toFixed(1) : '0.0'}%
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Latest Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {results.length > 0 ? results[results.length - 1].accuracy.toFixed(1) : '0.0'}%
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Improvement Runs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-blue-600">
              {results.filter(r => r.isImprovementRun).length}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Results Table */}
      <Card>
        <CardHeader>
          <CardTitle>Experiment Runs</CardTitle>
          <CardDescription>
            All prompt improvement experiments with calculated accuracy from actual results
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Run Name</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Accuracy</TableHead>
                <TableHead>True Decisions</TableHead>
                <TableHead>False Decisions</TableHead>
                <TableHead>Total Messages</TableHead>
                <TableHead>System Prompt Size</TableHead>
                <TableHead>Date</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {results.map((result, index) => (
                <TableRow key={result.filename}>
                  <TableCell className="font-medium">{result.runName}</TableCell>
                  <TableCell>
                    <Badge className={getRunTypeColor(result.isImprovementRun)}>
                      {result.isImprovementRun ? 'Improvement' : 'Original'}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge className={getAccuracyColor(result.accuracy)}>
                      {result.accuracy.toFixed(1)}%
                    </Badge>
                  </TableCell>
                  <TableCell className="text-green-600 font-semibold">
                    {result.trueDecisions}
                  </TableCell>
                  <TableCell className="text-red-600 font-semibold">
                    {result.falseDecisions}
                  </TableCell>
                  <TableCell>{result.totalMessages}</TableCell>
                  <TableCell>{result.systemPromptLength} chars</TableCell>
                  <TableCell className="text-sm text-gray-600">
                    {formatDate(result.processingDate)}
                  </TableCell>
                  <TableCell>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => setSelectedResult(result)}
                    >
                      Details
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Accuracy Trend */}
      {results.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle>Accuracy Trend</CardTitle>
            <CardDescription>
              Track accuracy improvements across iterations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {results.map((result, index) => (
                <div key={result.filename} className="flex items-center space-x-4">
                  <div className="w-20 text-sm text-gray-600">
                    {result.runName}
                  </div>
                  <div className="flex-1 bg-gray-200 rounded-full h-4">
                    <div 
                      className="bg-blue-600 h-4 rounded-full transition-all duration-300"
                      style={{ width: `${result.accuracy}%` }}
                    />
                  </div>
                  <div className="w-16 text-sm font-semibold">
                    {result.accuracy.toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Selected Result Details */}
      {selectedResult && (
        <Card>
          <CardHeader>
            <CardTitle>Run Details: {selectedResult.runName}</CardTitle>
            <CardDescription>
              Detailed information about the selected experiment run
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2">Performance Metrics</h4>
                <div className="space-y-2 text-sm">
                  <div>Accuracy: <span className="font-semibold">{selectedResult.accuracy.toFixed(2)}%</span></div>
                  <div>True Decisions: <span className="font-semibold text-green-600">{selectedResult.trueDecisions}</span></div>
                  <div>False Decisions: <span className="font-semibold text-red-600">{selectedResult.falseDecisions}</span></div>
                  <div>Total Messages: <span className="font-semibold">{selectedResult.totalMessages}</span></div>
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold mb-2">Run Information</h4>
                <div className="space-y-2 text-sm">
                  <div>Filename: <span className="font-mono text-xs">{selectedResult.filename}</span></div>
                  <div>Type: <span className={`px-2 py-1 rounded text-xs ${getRunTypeColor(selectedResult.isImprovementRun)}`}>
                    {selectedResult.isImprovementRun ? 'Improvement Run' : 'Original Run'}
                  </span></div>
                  <div>System Prompt: <span className="font-semibold">{selectedResult.systemPromptLength} characters</span></div>
                  <div>Date: <span className="font-semibold">{formatDate(selectedResult.processingDate)}</span></div>
                </div>
              </div>
            </div>
            
            <Separator className="my-4" />
            
            <div className="flex justify-end">
              <Button variant="outline" onClick={() => setSelectedResult(null)}>
                Close Details
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 