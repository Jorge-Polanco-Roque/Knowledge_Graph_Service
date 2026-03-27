import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useGraphStore } from '../../hooks/useGraphStore';

interface IngestButtonProps {
  userId: string;
}

interface FileInfo {
  name: string;
  size: number;
  extension: string;
}

interface FolderInfo {
  description: string;
  icon: string;
  files: FileInfo[];
  count: number;
}

interface FilesResponse {
  root_files: FileInfo[];
  folders: Record<string, FolderInfo>;
  total_files: number;
  path: string;
}

interface IngestResult {
  filename: string;
  folder: string;
  status: string;
  concepts_added: number;
  concept_labels?: string[];
}

export function IngestButton({ userId }: IngestButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [data, setData] = useState<FilesResponse | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [results, setResults] = useState<IngestResult[] | null>(null);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const setFullGraph = useGraphStore((s) => s.setFullGraph);
  const setStats = useGraphStore((s) => s.setStats);

  const loadFiles = async () => {
    try {
      const res = await fetch('/api/ingest/files');
      const d = await res.json();
      setData(d);
    } catch {
      setData(null);
    }
  };

  const handleToggle = () => {
    if (!isOpen) {
      loadFiles();
      setResults(null);
    }
    setIsOpen(!isOpen);
  };

  const toggleFolder = (name: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  };

  const handleIngestAll = async () => {
    setIsIngesting(true);
    setResults(null);
    try {
      const res = await fetch(`/api/ingest/${userId}`, { method: 'POST' });
      const d = await res.json();
      setResults(d.results || []);
      if (d.graph_snapshot) setFullGraph(d.graph_snapshot);
      if (d.stats) setStats(d.stats);
      const graphRes = await fetch(`/api/graph/${userId}`);
      if (graphRes.ok) setFullGraph(await graphRes.json());
      loadFiles();
    } catch (err) {
      console.error('Ingestion failed:', err);
    } finally {
      setIsIngesting(false);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const totalFiles = data?.total_files ?? 0;

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={handleToggle}
        style={{
          padding: '6px 12px',
          borderRadius: '8px',
          border: '1px solid rgba(249, 115, 22, 0.3)',
          background: isOpen ? 'rgba(249, 115, 22, 0.15)' : 'rgba(249, 115, 22, 0.08)',
          color: '#f97316',
          fontSize: '11px',
          fontWeight: 600,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '5px',
          transition: 'all 0.2s',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(249, 115, 22, 0.2)';
          e.currentTarget.style.boxShadow = '0 0 12px rgba(249, 115, 22, 0.2)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = isOpen ? 'rgba(249, 115, 22, 0.15)' : 'rgba(249, 115, 22, 0.08)';
          e.currentTarget.style.boxShadow = 'none';
        }}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        Ingest Files
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            style={{
              position: 'absolute',
              top: '100%',
              right: 0,
              marginTop: '8px',
              width: '360px',
              maxHeight: '500px',
              background: 'rgba(10, 10, 26, 0.95)',
              border: '1px solid rgba(249, 115, 22, 0.2)',
              borderRadius: '12px',
              padding: '14px',
              zIndex: 100,
              backdropFilter: 'blur(12px)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
              overflowY: 'auto',
            }}
          >
            <div style={{
              fontSize: '13px', fontWeight: 600, color: '#f97316',
              marginBottom: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              <span>📂 Input Folders</span>
              <span style={{ fontSize: '10px', color: '#8b949e', fontWeight: 400 }}>
                {totalFiles} files total
              </span>
            </div>

            {!data || totalFiles === 0 ? (
              <div style={{ fontSize: '12px', color: '#8b949e', padding: '16px 0', textAlign: 'center' }}>
                <div style={{ marginBottom: '8px' }}>No files found.</div>
                <div style={{ fontSize: '11px', color: '#6b7280' }}>
                  Drop files into subfolders of<br />
                  <code style={{ color: '#f97316', fontSize: '10px' }}>backend/inputs/</code>
                </div>
                <div style={{
                  marginTop: '12px', fontSize: '10px', color: '#4a5568',
                  display: 'flex', flexWrap: 'wrap', gap: '4px', justifyContent: 'center',
                }}>
                  {['📄 documents', '👤 people', '🚀 projects', '📚 knowledge', '💻 code', '💬 conversations', '📊 data', '🖼️ media'].map((f) => (
                    <span key={f} style={{
                      padding: '1px 6px', borderRadius: '6px',
                      background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
                    }}>
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <>
                <div style={{ marginBottom: '10px' }}>
                  {/* Root files */}
                  {data.root_files.length > 0 && (
                    <FolderSection
                      name="root"
                      icon="📁"
                      description="Root level files"
                      files={data.root_files}
                      expanded={expandedFolders.has('root')}
                      onToggle={() => toggleFolder('root')}
                      formatSize={formatSize}
                    />
                  )}

                  {/* Subfolders */}
                  {Object.entries(data.folders).map(([name, folder]) => (
                    <FolderSection
                      key={name}
                      name={name}
                      icon={folder.icon}
                      description={folder.description}
                      files={folder.files}
                      expanded={expandedFolders.has(name)}
                      onToggle={() => toggleFolder(name)}
                      formatSize={formatSize}
                    />
                  ))}
                </div>

                <button
                  onClick={handleIngestAll}
                  disabled={isIngesting}
                  style={{
                    width: '100%',
                    padding: '9px',
                    borderRadius: '8px',
                    border: 'none',
                    background: isIngesting
                      ? 'rgba(249, 115, 22, 0.2)'
                      : 'linear-gradient(135deg, #f97316, #f59e0b)',
                    color: '#fff',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: isIngesting ? 'wait' : 'pointer',
                    transition: 'all 0.2s',
                  }}
                >
                  {isIngesting ? (
                    <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                      <motion.span
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                        style={{ display: 'inline-block' }}
                      >
                        ⚙️
                      </motion.span>
                      Processing all folders...
                    </span>
                  ) : (
                    `Ingest All (${totalFiles} files)`
                  )}
                </button>
              </>
            )}

            {/* Results */}
            {results && results.length > 0 && (
              <div style={{ marginTop: '10px', borderTop: '1px solid rgba(249, 115, 22, 0.15)', paddingTop: '10px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#10b981', marginBottom: '6px' }}>
                  Ingestion Results
                </div>
                {results.map((r, i) => (
                  <div
                    key={`${r.filename}-${i}`}
                    style={{
                      fontSize: '11px',
                      padding: '3px 0',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      color: r.status === 'ingested' ? '#e6edf3' : '#8b949e',
                    }}
                  >
                    <span style={{
                      overflow: 'hidden', textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap', maxWidth: '150px',
                    }}>
                      {r.filename}
                    </span>
                    <span style={{ fontSize: '10px', color: '#6b7280' }}>{r.folder}</span>
                    <span style={{
                      color: r.status === 'ingested' ? '#10b981' : '#f97316',
                      fontWeight: 500, flexShrink: 0,
                    }}>
                      {r.status === 'ingested' ? `+${r.concepts_added}` : r.status}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function FolderSection({
  name, icon, description, files, expanded, onToggle, formatSize,
}: {
  name: string;
  icon: string;
  description: string;
  files: FileInfo[];
  expanded: boolean;
  onToggle: () => void;
  formatSize: (n: number) => string;
}) {
  if (files.length === 0) return null;

  return (
    <div style={{ marginBottom: '4px' }}>
      <button
        onClick={onToggle}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '6px 8px',
          borderRadius: '6px',
          border: 'none',
          background: expanded ? 'rgba(249, 115, 22, 0.08)' : 'rgba(255, 255, 255, 0.02)',
          cursor: 'pointer',
          color: '#e6edf3',
          fontSize: '12px',
          transition: 'background 0.15s',
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ fontSize: '13px' }}>{icon}</span>
          <span style={{ fontWeight: 600 }}>{name}</span>
          <span style={{ color: '#8b949e', fontSize: '10px', fontWeight: 400 }}>
            {files.length} {files.length === 1 ? 'file' : 'files'}
          </span>
        </span>
        <span style={{ color: '#8b949e', fontSize: '10px', transform: expanded ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>
          ▶
        </span>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{ overflow: 'hidden', paddingLeft: '24px' }}
          >
            {files.map((file) => (
              <div
                key={file.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '3px 6px',
                  fontSize: '11px',
                }}
              >
                <span style={{
                  color: '#c9d1d9',
                  overflow: 'hidden', textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap', maxWidth: '220px',
                }}>
                  {file.name}
                </span>
                <span style={{ color: '#6b7280', fontSize: '10px', flexShrink: 0 }}>
                  {formatSize(file.size)}
                </span>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
