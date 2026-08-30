import axios from 'axios'

const api = axios.create({ baseURL: '/' })

export interface SemanticEvaluation {
  is_correct?: boolean | null
  score?: number | null
  reason?: string
  issues?: string[]
}

export interface ResourceGuard {
  decision?: string
  risk_level?: string
  violations?: string[]
  reason?: string
}

export interface Confidence {
  score?: number
  level?: string
  factors?: Record<string, number>
}

export interface Explanation {
  summary?: string
  tables_used?: string[]
  operation_count?: number
}

export interface Visualization {
  recommended?: boolean
  chart_type?: string | null
  x_axis?: string | null
  y_axis?: string | null
  reason?: string
  chart?: {
    rendered?: boolean
    chart_type?: string | null
    image_base64?: string
  }
}

export interface QueryResponse {
  sql: string
  results: Record<string, unknown>[]
  tables_used: string[]
  requires_approval: boolean
  approval_reason?: string

  semantic_evaluation?: SemanticEvaluation
  explanation?: Explanation
  visualization?: Visualization
  resource_guard?: ResourceGuard
  confidence?: Confidence

  cache?: {
    hit?: boolean
  }

  error?: string
  latency_ms: number
}

export interface ApproveResponse {
  executed: boolean
  results: Record<string, unknown>[]
  message: string
}

export interface SchemaColumn {
  name: string
  description: string
}

export interface SchemaTable {
  table_name: string
  description: string
  columns: SchemaColumn[]
}

export async function postQuery(question: string): Promise<QueryResponse> {
  const { data } = await api.post<QueryResponse>('/api/query', { question })
  return data
}

export async function postApprove(
  sql: string,
  approved: boolean
): Promise<ApproveResponse> {
  const { data } = await api.post<ApproveResponse>('/api/approve', {
    sql,
    approved,
  })
  return data
}

export async function getSchema(): Promise<SchemaTable[]> {
  const { data } = await api.get<SchemaTable[]>('/api/schema')
  return data
}

export async function getHealth(): Promise<Record<string, unknown>> {
  const { data } = await api.get<Record<string, unknown>>('/api/health')
  return data
}