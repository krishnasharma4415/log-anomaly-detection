import { Cpu, Brain, Zap, GitMerge } from 'lucide-react';

// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';
export const API_ENDPOINTS = {
  HEALTH: '/health',
  MODEL_INFO: '/model-info',
  PREDICT: '/api/predict'
};

// Model Options
export const MODEL_OPTIONS = [
  {
    id: 'ml',
    name: 'ML Model',
    icon: Cpu,
    description: 'Traditional Machine Learning',
    color: 'blue'
  },
  {
    id: 'dann_bert',
    name: 'DANN BERT',
    icon: Brain,
    description: 'Domain Adversarial Neural Network',
    color: 'purple'
  },
  {
    id: 'lora_bert',
    name: 'LoRA BERT',
    icon: Zap,
    description: 'Low-Rank Adaptation',
    color: 'pink'
  },
  {
    id: 'hybrid_bert',
    name: 'Hybrid BERT',
    icon: GitMerge,
    description: 'Combined Approach',
    color: 'green'
  }
];

// Supported Log Sources
export const LOG_SOURCES = [
  'Windows', 'Linux', 'Mac', 'Hadoop', 'HDFS', 'Zookeeper',
  'Spark', 'Apache', 'Thunderbird', 'Proxifier', 'HealthApp',
  'OpenStack', 'OpenSSH', 'BGL', 'HPC', 'Android'
];

// Log Type Colors
export const LOG_TYPE_COLORS = {
  'OpenSSH': 'bg-blue-500/20 border-blue-500/30 text-blue-300',
  'Apache': 'bg-red-500/20 border-red-500/30 text-red-300',
  'HDFS': 'bg-green-500/20 border-green-500/30 text-green-300',
  'Hadoop': 'bg-yellow-500/20 border-yellow-500/30 text-yellow-300',
  'Linux': 'bg-purple-500/20 border-purple-500/30 text-purple-300',
  'Windows': 'bg-cyan-500/20 border-cyan-500/30 text-cyan-300',
  'Spark': 'bg-orange-500/20 border-orange-500/30 text-orange-300',
  'Android': 'bg-lime-500/20 border-lime-500/30 text-lime-300',
  'BGL': 'bg-indigo-500/20 border-indigo-500/30 text-indigo-300',
  'Mac': 'bg-pink-500/20 border-pink-500/30 text-pink-300',
  'OpenStack': 'bg-teal-500/20 border-teal-500/30 text-teal-300',
  'Zookeeper': 'bg-amber-500/20 border-amber-500/30 text-amber-300',
  'Unknown': 'bg-gray-500/20 border-gray-500/30 text-gray-300'
};

// Sample Log Data
export const SAMPLE_LOG = `2025-01-15 14:32:15 ERROR Connection timeout to server 192.168.1.100
2025-01-15 14:32:20 WARN Retrying connection attempt 3/5
2025-01-15 14:32:25 ERROR Failed to establish connection after 5 attempts
2025-01-15 14:32:30 INFO Switching to backup server 192.168.1.101
2025-01-15 14:32:35 INFO Connection established successfully
2025-01-15 14:32:40 DEBUG Processing request queue (125 pending)
2025-01-15 14:32:45 INFO Request processed successfully
2025-01-15 14:32:50 ERROR Unexpected null pointer exception in module auth.service
2025-01-15 14:32:55 CRITICAL System memory usage at 95%
2025-01-15 14:33:00 WARN High CPU usage detected (88%)`;

// UI Configuration
export const UI_CONFIG = {
  MAX_DETAILED_RESULTS: 10,
  RESULTS_MAX_HEIGHT: '600px',
  DETAILED_RESULTS_MAX_HEIGHT: '500px'
};

// Model Type Descriptions
export const MODEL_DESCRIPTIONS = {
  'DANN-BERT': 'Domain Adversarial Neural Network with BERT for cross-domain log analysis.',
  'LORA-BERT': 'Low-Rank Adaptation of BERT for efficient parameter-tuning.',
  'HYBRID-BERT': 'Combines BERT embeddings with template features for enhanced accuracy.',
  'ML Model': 'Traditional machine learning with advanced feature engineering.'
};