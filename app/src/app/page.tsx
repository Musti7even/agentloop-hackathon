'use client'

import { useState, useEffect } from 'react'
import { ChevronDownIcon, PlayIcon, CheckCircleIcon, XCircleIcon, ArrowTrendingUpIcon, ArrowTrendingDownIcon } from '@heroicons/react/24/outline'

interface AnalyzedResult {
  fileName: string;
  runName: string;
  accuracy: number;
  trueDecisions: number;
  falseDecisions: number;
  totalMessages: number;
  totalPersonas: number;
  isImprovementRun: boolean;
  processingDate: string;
  systemPrompt: string;
}

export default function Home() {
  const [results, setResults] = useState<AnalyzedResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<AnalyzedResult | null>(null);

  useEffect(() => {
    fetchResults();
  }, []);

  const fetchResults = async () => {
    try {
      const response = await fetch('/api/results');
      if (!response.ok) {
        throw new Error('Failed to fetch results');
      }
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 90) return 'text-emerald-400';
    if (accuracy >= 75) return 'text-yellow-400';
    if (accuracy >= 60) return 'text-orange-400';
    return 'text-red-400';
  };

  const getAccuracyBgColor = (accuracy: number) => {
    if (accuracy >= 90) return 'bg-emerald-500/10 border-emerald-500/20';
    if (accuracy >= 75) return 'bg-yellow-500/10 border-yellow-500/20';
    if (accuracy >= 60) return 'bg-orange-500/10 border-orange-500/20';
    return 'bg-red-500/10 border-red-500/20';
  };

  const bestAccuracy = results.length > 0 ? Math.max(...results.map(r => r.accuracy)) : 0;
  const latestAccuracy = results.length > 0 ? results[0].accuracy : 0;
  const improvementRuns = results.filter(r => r.isImprovementRun).length;
  const totalMessages = results.reduce((sum, r) => sum + r.totalMessages, 0);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="text-gray-300">Loading results...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <XCircleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-300 mb-2">Error Loading Results</h2>
          <p className="text-gray-500">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white mb-1">
                Prompt Improvement Dashboard
              </h1>
              <p className="text-gray-400">
                Monitor and analyze prompt performance across experiments
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <div className="px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full">
                <span className="text-blue-400 text-sm font-medium">{results.length} Experiments</span>
              </div>
              <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors">
                New Experiment
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 backdrop-blur">
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <PlayIcon className="h-5 w-5 text-blue-400" />
              </div>
            </div>
            <div className="text-2xl font-bold text-white mb-1">{results.length}</div>
            <div className="text-sm text-gray-400">Total Experiments</div>
          </div>
          
                     <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 backdrop-blur">
             <div className="flex items-center justify-between mb-4">
               <div className="p-2 bg-emerald-500/10 rounded-lg">
                 <ArrowTrendingUpIcon className="h-5 w-5 text-emerald-400" />
               </div>
               <div className={`text-xs px-2 py-1 rounded-full ${getAccuracyBgColor(bestAccuracy)}`}>
                 Best
               </div>
             </div>
            <div className={`text-2xl font-bold mb-1 ${getAccuracyColor(bestAccuracy)}`}>
              {bestAccuracy.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">Peak Accuracy</div>
          </div>
          
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 backdrop-blur">
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-purple-500/10 rounded-lg">
                <CheckCircleIcon className="h-5 w-5 text-purple-400" />
              </div>
                             {latestAccuracy > bestAccuracy * 0.9 ? (
                 <ArrowTrendingUpIcon className="h-4 w-4 text-emerald-400" />
               ) : (
                 <ArrowTrendingDownIcon className="h-4 w-4 text-red-400" />
               )}
            </div>
            <div className={`text-2xl font-bold mb-1 ${getAccuracyColor(latestAccuracy)}`}>
              {latestAccuracy.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">Latest Accuracy</div>
          </div>
          
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 backdrop-blur">
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-orange-500/10 rounded-lg">
                <svg className="h-5 w-5 text-orange-400" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
            <div className="text-2xl font-bold text-white mb-1">{totalMessages.toLocaleString()}</div>
            <div className="text-sm text-gray-400">Total Messages</div>
          </div>
        </div>

        {/* Experiments Table */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl backdrop-blur">
          <div className="px-6 py-4 border-b border-gray-800">
            <h2 className="text-lg font-semibold text-white">Experiments</h2>
            <p className="text-sm text-gray-400 mt-1">
              Performance metrics calculated from actual decision outcomes
            </p>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left text-xs font-medium text-gray-400 uppercase tracking-wider px-6 py-4">
                    Experiment
                  </th>
                  <th className="text-left text-xs font-medium text-gray-400 uppercase tracking-wider px-6 py-4">
                    Type
                  </th>
                  <th className="text-left text-xs font-medium text-gray-400 uppercase tracking-wider px-6 py-4">
                    Accuracy
                  </th>
                  <th className="text-left text-xs font-medium text-gray-400 uppercase tracking-wider px-6 py-4">
                    Results
                  </th>
                  <th className="text-left text-xs font-medium text-gray-400 uppercase tracking-wider px-6 py-4">
                    Date
                  </th>
                  <th className="text-left text-xs font-medium text-gray-400 uppercase tracking-wider px-6 py-4">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {results.map((result, index) => (
                  <tr key={index} className="hover:bg-gray-800/50 transition-colors">
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-3">
                        <div className={`w-2 h-2 rounded-full ${
                          result.isImprovementRun ? 'bg-blue-500' : 'bg-gray-500'
                        }`} />
                        <span className="text-white font-medium">{result.runName}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                        result.isImprovementRun 
                          ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' 
                          : 'bg-gray-500/10 text-gray-400 border border-gray-500/20'
                      }`}>
                        {result.isImprovementRun ? 'Improvement' : 'Baseline'}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        <span className={`text-lg font-semibold ${getAccuracyColor(result.accuracy)}`}>
                          {result.accuracy.toFixed(1)}%
                        </span>
                        <div className="w-16 bg-gray-800 rounded-full h-1.5">
                          <div 
                            className="bg-gradient-to-r from-blue-500 to-purple-600 h-1.5 rounded-full transition-all"
                            style={{ width: `${result.accuracy}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-4 text-sm">
                        <div className="flex items-center space-x-1">
                          <CheckCircleIcon className="h-4 w-4 text-emerald-400" />
                          <span className="text-emerald-400 font-medium">{result.trueDecisions}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <XCircleIcon className="h-4 w-4 text-red-400" />
                          <span className="text-red-400 font-medium">{result.falseDecisions}</span>
                        </div>
                        <span className="text-gray-400">/ {result.totalMessages}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-gray-400 text-sm">
                        {new Date(result.processingDate).toLocaleDateString()}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <button 
                        onClick={() => setSelectedRun(result)}
                        className="text-blue-400 hover:text-blue-300 text-sm font-medium transition-colors"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Performance Chart */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 mt-8 backdrop-blur">
          <h2 className="text-lg font-semibold text-white mb-6">Performance Trend</h2>
          <div className="space-y-4">
            {results.slice().reverse().map((result, index) => (
              <div key={index} className="flex items-center space-x-4">
                <div className="w-32 text-sm text-gray-400 font-medium">
                  {result.runName}
                </div>
                <div className="flex-1 bg-gray-800 rounded-full h-3 relative overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-blue-500 via-purple-500 to-emerald-500 h-3 rounded-full transition-all duration-1000 ease-out"
                    style={{ width: `${result.accuracy}%` }}
                  />
                </div>
                <div className={`w-16 text-sm font-semibold ${getAccuracyColor(result.accuracy)}`}>
                  {result.accuracy.toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Detail Modal */}
      {selectedRun && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur flex items-center justify-center p-4 z-50">
          <div className="bg-gray-900 border border-gray-800 rounded-xl max-w-4xl w-full max-h-[80vh] overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">{selectedRun.runName} Details</h3>
              <button 
                onClick={() => setSelectedRun(null)}
                className="text-gray-400 hover:text-gray-300"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-6 overflow-y-auto max-h-[calc(80vh-80px)]">
              <div className="grid grid-cols-2 gap-6 mb-6">
                <div>
                  <h4 className="text-sm font-medium text-gray-400 mb-2">Performance Metrics</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Accuracy:</span>
                      <span className={`font-semibold ${getAccuracyColor(selectedRun.accuracy)}`}>
                        {selectedRun.accuracy.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">True Decisions:</span>
                      <span className="text-emerald-400 font-semibold">{selectedRun.trueDecisions}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">False Decisions:</span>
                      <span className="text-red-400 font-semibold">{selectedRun.falseDecisions}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Total Messages:</span>
                      <span className="text-white font-semibold">{selectedRun.totalMessages}</span>
                    </div>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-400 mb-2">Experiment Info</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Type:</span>
                      <span className={`font-semibold ${
                        selectedRun.isImprovementRun ? 'text-blue-400' : 'text-gray-400'
                      }`}>
                        {selectedRun.isImprovementRun ? 'Improvement' : 'Baseline'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Personas:</span>
                      <span className="text-white font-semibold">{selectedRun.totalPersonas}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Date:</span>
                      <span className="text-white font-semibold">
                        {new Date(selectedRun.processingDate).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              {selectedRun.systemPrompt && (
                <div>
                  <h4 className="text-sm font-medium text-gray-400 mb-3">System Prompt</h4>
                  <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
                    <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono">
                      {selectedRun.systemPrompt}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
} 