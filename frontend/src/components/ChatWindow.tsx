import { useState, useRef, useEffect } from 'react'
import { postQuery, type QueryResponse } from '../api'
import SqlDisplay from './SqlDisplay'
import ResultsTable from './ResultsTable'
import type { Message } from '../App'

interface Props {
  messages: Message[]
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  onAnswer: (response: QueryResponse) => void
}

function ResponseMessage({ response }: { response: QueryResponse }) {
  const guard = response.resource_guard
  const semantic = response.semantic_evaluation
  const confidence = response.confidence
  const explanation = response.explanation

  const isBlocked = guard?.decision === 'BLOCK'

  const isSecurityBlocked =
    isBlocked &&
    guard?.violations?.some(
      (v) =>
        v === 'USER_INPUT_SECURITY' ||
        v === 'SQL_SAFETY'
    )

  const isEmptySql =
    guard?.violations?.includes('EMPTY_SQL') ||
    (!response.sql && !isSecurityBlocked)

  const isIncomplete =
    semantic?.is_correct === false

  // ------------------------------------------------------------
  // SECURITY BLOCK
  // ------------------------------------------------------------
  if (isSecurityBlocked) {
    return (
      <div style={{
        background: 'rgba(255, 85, 85, 0.08)',
        border: '1px solid var(--accent-red)',
        borderRadius: '8px',
        padding: '16px',
        fontFamily: 'var(--font-mono)',
      }}>
        <div style={{
          color: 'var(--accent-red)',
          fontSize: '14px',
          fontWeight: 700,
          marginBottom: '8px',
        }}>
          🛡 Request blocked for security
        </div>

        <div style={{
          color: 'var(--text-primary)',
          fontSize: '12px',
          lineHeight: 1.6,
        }}>
          This request contains a potentially destructive database
          operation, so it was blocked before any database operation
          was performed.
        </div>

        <div style={{
          color: 'var(--accent-green)',
          fontSize: '12px',
          marginTop: '10px',
          fontWeight: 600,
        }}>
          ✓ Your data was not changed.
        </div>

        {guard?.reason && (
          <div style={{
            color: 'var(--text-secondary)',
            fontSize: '11px',
            marginTop: '10px',
          }}>
            {guard.reason}
          </div>
        )}

        <div style={{
          color: 'var(--text-secondary)',
          fontSize: '11px',
          marginTop: '12px',
        }}>
          Try asking a read-only question about orders, customers,
          products, sellers, reviews, or revenue.
        </div>
      </div>
    )
  }

  // ------------------------------------------------------------
  // OUTSIDE DATABASE SCOPE / EMPTY SQL
  // ------------------------------------------------------------
  if (isEmptySql) {
    return (
      <div style={{
        background: 'rgba(255, 193, 7, 0.06)',
        border: '1px solid rgba(255, 193, 7, 0.45)',
        borderRadius: '8px',
        padding: '16px',
        fontFamily: 'var(--font-mono)',
      }}>
        <div style={{
          color: '#ffc107',
          fontSize: '14px',
          fontWeight: 700,
          marginBottom: '8px',
        }}>
          ℹ Question outside available data
        </div>

        <div style={{
          color: 'var(--text-primary)',
          fontSize: '12px',
          lineHeight: 1.6,
        }}>
          I couldn't find information in the available Olist database
          that answers this question.
        </div>

        <div style={{
          color: 'var(--text-secondary)',
          fontSize: '11px',
          marginTop: '10px',
          lineHeight: 1.6,
        }}>
          You can ask about:
          <br />
          <strong style={{ color: 'var(--accent-cyan)' }}>
            Orders · Customers · Products · Sellers · Reviews · Revenue
          </strong>
        </div>
      </div>
    )
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '12px',
    }}>

      {/* ------------------------------------------------------ */}
      {/* INCOMPLETE / LOW CONFIDENCE */}
      {/* ------------------------------------------------------ */}
      {isIncomplete && (
        <div style={{
          background: 'rgba(255, 193, 7, 0.06)',
          border: '1px solid rgba(255, 193, 7, 0.45)',
          borderRadius: '8px',
          padding: '14px 16px',
          fontFamily: 'var(--font-mono)',
        }}>
          <div style={{
            color: '#ffc107',
            fontSize: '13px',
            fontWeight: 700,
            marginBottom: '6px',
          }}>
            ⚠ Answer may be incomplete
          </div>

          <div style={{
            color: 'var(--text-primary)',
            fontSize: '12px',
            lineHeight: 1.6,
          }}>
            {semantic?.reason ||
              'The generated query did not completely answer your question.'}
          </div>

          {semantic?.issues && semantic.issues.length > 0 && (
            <ul style={{
              margin: '8px 0 0 18px',
              padding: 0,
              color: 'var(--text-secondary)',
              fontSize: '11px',
              lineHeight: 1.5,
            }}>
              {semantic.issues.map((issue, index) => (
                <li key={index}>{issue}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* ------------------------------------------------------ */}
      {/* INSIGHT */}
      {/* ------------------------------------------------------ */}
      {explanation?.summary && (
        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          padding: '14px 16px',
        }}>
          <div style={{
            color: 'var(--accent-green)',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            fontWeight: 700,
            letterSpacing: '0.08em',
            marginBottom: '7px',
          }}>
            INSIGHT
          </div>

          <div style={{
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-mono)',
            fontSize: '13px',
            lineHeight: 1.6,
          }}>
            {explanation.summary}
          </div>
        </div>
      )}

      {/* ------------------------------------------------------ */}
      {/* RESULTS */}
      {/* ------------------------------------------------------ */}
      {response.results.length > 0 && (
        <div>
          <div style={{
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: '10px',
            marginBottom: '6px',
            letterSpacing: '0.06em',
          }}>
            RESULTS · {response.results.length} ROW
            {response.results.length !== 1 ? 'S' : ''}
          </div>

          <ResultsTable results={response.results} />
        </div>
      )}

      {/* ------------------------------------------------------ */}
      {/* SQL */}
      {/* ------------------------------------------------------ */}
      {response.sql && (
        <SqlDisplay
          sql={response.sql}
          latencyMs={response.latency_ms}
          requiresApproval={response.requires_approval}
          approvalReason={response.approval_reason}
        />
      )}

      {/* ------------------------------------------------------ */}
      {/* CONFIDENCE */}
      {/* ------------------------------------------------------ */}
      {confidence?.score !== undefined && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          color: 'var(--text-secondary)',
          fontFamily: 'var(--font-mono)',
          fontSize: '10px',
        }}>
          <span>CONFIDENCE</span>
          <span style={{
            color:
              confidence.level === 'HIGH'
                ? 'var(--accent-green)'
                : confidence.level === 'MEDIUM'
                  ? '#ffc107'
                  : 'var(--accent-red)',
            fontWeight: 700,
          }}>
            {confidence.score.toFixed(0)}%
          </span>
          <span>·</span>
          <span>{confidence.level || 'UNKNOWN'}</span>
        </div>
      )}

      {/* ------------------------------------------------------ */}
      {/* APPROVAL */}
      {/* ------------------------------------------------------ */}
      {response.requires_approval && (
        <div style={{
          color: '#ffc107',
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
        }}>
          ⚠ Approval required before this query can execute.
        </div>
      )}

      {/* ------------------------------------------------------ */}
      {/* FALLBACK */}
      {/* ------------------------------------------------------ */}
      {response.results.length === 0 &&
        !response.requires_approval &&
        !isIncomplete &&
        !explanation?.summary && (
          <div style={{
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            padding: '10px 12px',
          }}>
            The query completed, but returned no rows.
          </div>
        )}
    </div>
  )
}

export default function ChatWindow({
  messages,
  setMessages,
  onAnswer,
}: Props) {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async () => {
    const q = question.trim()
    if (!q || loading) return

    const questionMsg: Message = {
      id: crypto.randomUUID(),
      type: 'question',
      content: q,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, questionMsg])
    setQuestion('')
    setLoading(true)

    try {
      const response = await postQuery(q)
      onAnswer(response)
    } catch (err: unknown) {
      const errMsg =
        err instanceof Error ? err.message : String(err)

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          type: 'error',
          content: errMsg,
          timestamp: new Date(),
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (
    e: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
    }}>

      {/* Message history */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
      }}>

        {messages.length === 0 && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: '13px',
            gap: '8px',
          }}>
            <div style={{ fontSize: '32px' }}>⚡</div>

            <div>
              Ask a question about your Olist data
            </div>

            <div style={{
              fontSize: '11px',
              opacity: 0.6,
            }}>
              e.g. "What is the total revenue by product category?"
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id}>

            {msg.type === 'question' && (
              <div style={{
                display: 'flex',
                justifyContent: 'flex-end',
              }}>
                <div style={{
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--accent-cyan)',
                  borderRadius: '8px 8px 2px 8px',
                  padding: '10px 14px',
                  maxWidth: '70%',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '13px',
                  color: 'var(--accent-cyan)',
                }}>
                  {msg.content as string}
                </div>
              </div>
            )}

            {msg.type === 'answer' && (
              <ResponseMessage
                response={msg.content as QueryResponse}
              />
            )}

            {msg.type === 'error' && (
              <div style={{
                background: 'rgba(255,85,85,0.1)',
                border: '1px solid var(--accent-red)',
                borderRadius: '6px',
                padding: '10px 14px',
                fontFamily: 'var(--font-mono)',
                fontSize: '12px',
                color: 'var(--accent-red)',
              }}>
                ⚠ {msg.content as string}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
          }}>
            <span style={{
              animation: 'pulse 1.2s infinite',
            }}>
              ◉
            </span>

            <span>
              Analyzing your question...
            </span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{
        borderTop: '1px solid var(--border-color)',
        padding: '16px',
        background: 'var(--bg-secondary)',
      }}>
        <div style={{
          display: 'flex',
          gap: '10px',
          alignItems: 'flex-end',
        }}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your data..."
            disabled={loading}
            style={{
              flex: 1,
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '10px 12px',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-mono)',
              fontSize: '13px',
              resize: 'none',
              minHeight: '44px',
              maxHeight: '120px',
              outline: 'none',
              lineHeight: 1.5,
            }}
            rows={1}
          />

          <button
            onClick={handleSubmit}
            disabled={loading || !question.trim()}
            style={{
              background:
                loading || !question.trim()
                  ? 'var(--bg-tertiary)'
                  : 'var(--accent-green)',
              color:
                loading || !question.trim()
                  ? 'var(--text-secondary)'
                  : '#0d1117',
              border: 'none',
              borderRadius: '6px',
              padding: '10px 18px',
              fontWeight: 600,
              fontSize: '13px',
              transition: 'all 0.15s',
              whiteSpace: 'nowrap',
            }}
          >
            {loading ? '...' : 'Run ↵'}
          </button>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  )
}