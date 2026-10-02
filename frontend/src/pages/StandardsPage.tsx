import { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  AlertTriangle,
  ArrowRight,
  BookMarked,
  Check,
  CheckCircle2,
  ChevronRight,
  Columns,
  ExternalLink,
  FileCheck2,
  FileText,
  FileWarning,
  Filter,
  History,
  Layers,
  Scale,
  Search,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  X,
} from 'lucide-react';
import { TopNav } from '@/components/TopNav';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { useRouter } from '@/router';
import { standards, statusConfig, procurementCategories } from '@/data/mockData';
import type { Standard, StandardRelationshipRole, StandardStatus } from '@/data/types';
import { StandardComparisonModal } from '@/components/standards/StandardComparisonModal';
import { adaptStandard } from '@/services/adapter';
import { listStandards, searchStandards } from '@/services/api';

const toStandardArray = (res: any): any[] =>
  Array.isArray(res) ? res : (res?.items || res?.results || res?.standards || res?.data || []);

export function StandardsPage() {
  const { navigate } = useRouter();
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<StandardStatus | 'all'>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [relationshipFilter, setRelationshipFilter] = useState<StandardRelationshipRole | 'all'>('all');
  const [sortBy, setSortBy] = useState<'relevance' | 'current-first' | 'recent-update'>('relevance');

  // Comparison selection
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);
  const [isCompareModalOpen, setIsCompareModalOpen] = useState(false);

  // Real BIS catalog from the live backend (falls back to the seeded demo set until loaded/if offline)
  const [baseStandards, setBaseStandards] = useState<Standard[]>(standards);
  const [searchResults, setSearchResults] = useState<Standard[] | null>(null);

  useEffect(() => {
    let alive = true;
    listStandards(0, 48)
      .then((res) => {
        const rows = toStandardArray(res).map(adaptStandard);
        if (alive && rows.length) setBaseStandards(rows);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const q = search.trim();
    if (q.length < 2) {
      setSearchResults(null);
      return;
    }
    let alive = true;
    const t = setTimeout(() => {
      searchStandards(q)
        .then((res) => {
          if (alive) setSearchResults(toStandardArray(res).map(adaptStandard));
        })
        .catch(() => {
          if (alive) setSearchResults(null);
        });
    }, 300);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [search]);

  const toggleCompare = (id: string) => {
    setSelectedForCompare((prev) => {
      if (prev.includes(id)) {
        return prev.filter((item) => item !== id);
      }
      if (prev.length >= 2) {
        return [prev[1], id];
      }
      return [...prev, id];
    });
  };

  // Filter and sort standards
  const filteredStandards = useMemo(() => {
    const usingServerSearch = searchResults !== null;
    const source = searchResults ?? baseStandards;
    return source
      .filter((s) => {
        // Search query (client-side only when not already server-filtered)
        if (search && !usingServerSearch) {
          const q = search.toLowerCase();
          const matchNumber = s.number.toLowerCase().includes(q);
          const matchTitle = s.title.toLowerCase().includes(q);
          const matchSummary = s.summary.toLowerCase().includes(q);
          const matchKeywords = s.keywords.some((k) => k.toLowerCase().includes(q));
          const matchCoverage = s.technicalCoverage ? s.technicalCoverage.toLowerCase().includes(q) : false;
          if (!matchNumber && !matchTitle && !matchSummary && !matchKeywords && !matchCoverage) {
            return false;
          }
        }

        // Status filter
        if (statusFilter !== 'all' && s.status !== statusFilter) {
          return false;
        }

        // Category filter
        if (categoryFilter !== 'all') {
          if (s.category && s.category.toLowerCase() !== categoryFilter.toLowerCase()) {
            return false;
          }
        }

        // Relationship filter
        if (relationshipFilter !== 'all' && s.relationshipRole !== relationshipFilter) {
          return false;
        }

        return true;
      })
      .sort((a, b) => {
        if (sortBy === 'current-first') {
          if (a.status === 'current' && b.status !== 'current') return -1;
          if (b.status === 'current' && a.status !== 'current') return 1;
          return (b.applicabilityScore || 0) - (a.applicabilityScore || 0);
        }
        if (sortBy === 'recent-update') {
          return (b.lastUpdatedDate || '2000').localeCompare(a.lastUpdatedDate || '2000');
        }
        // Default: relevance / applicability score
        return (b.applicabilityScore || 0) - (a.applicabilityScore || 0);
      });
  }, [search, statusFilter, categoryFilter, relationshipFilter, sortBy, baseStandards, searchResults]);

  // Featured / Recently Relevant items for research intelligence strip
  const featuredStandards = useMemo(() => {
    return baseStandards
      .filter((s) => s.applicabilityScore && s.applicabilityScore >= 80)
      .slice(0, 3);
  }, [baseStandards]);

  // Relationship badge helper
  const renderRoleBadge = (role?: StandardRelationshipRole) => {
    switch (role) {
      case 'primary':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-teal-800 text-white px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase font-mono shadow-soft">
            Primary Code
          </span>
        );
      case 'normative':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-blue-50 text-blue-800 border border-blue-200 px-2 py-0.5 text-[10px] font-medium font-mono">
            Normative Reference
          </span>
        );
      case 'testing':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-purple-50 text-purple-800 border border-purple-200 px-2 py-0.5 text-[10px] font-medium font-mono">
            Testing Protocol
          </span>
        );
      case 'safety':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-amber-50 text-amber-900 border border-amber-200 px-2 py-0.5 text-[10px] font-medium font-mono">
            Safety Standard
          </span>
        );
      case 'installation':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 px-2 py-0.5 text-[10px] font-medium font-mono">
            Design & Installation
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 rounded bg-ink-100 text-ink-700 border border-ink-200 px-2 py-0.5 text-[10px] font-medium font-mono">
            Related
          </span>
        );
    }
  };

  // Status badge with explicit text labels
  const renderStatusBadge = (status: StandardStatus) => {
    switch (status) {
      case 'current':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-success-50 text-success-800 border border-success-200 px-2 py-0.5 text-[10px] font-semibold font-mono uppercase tracking-wider">
            <CheckCircle2 size={10} className="text-success-600" />
            CURRENT
          </span>
        );
      case 'amended':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-teal-50 text-teal-800 border border-teal-200 px-2 py-0.5 text-[10px] font-semibold font-mono uppercase tracking-wider">
            AMENDED
          </span>
        );
      case 'under-review':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-warning-50 text-warning-800 border border-warning-200 px-2 py-0.5 text-[10px] font-semibold font-mono uppercase tracking-wider">
            <AlertTriangle size={10} className="text-warning-600" />
            NEEDS REVIEW
          </span>
        );
      case 'superseded':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-error-50 text-error-800 border border-error-200 px-2 py-0.5 text-[10px] font-semibold font-mono uppercase tracking-wider">
            SUPERSEDED
          </span>
        );
      case 'withdrawn':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-ink-100 text-ink-800 border border-ink-200 px-2 py-0.5 text-[10px] font-semibold font-mono uppercase tracking-wider">
            WITHDRAWN
          </span>
        );
    }
  };

  return (
    <div className="min-h-screen bg-ivory-50 text-ink-900 pb-24 dark:bg-[#090D16] dark:text-slate-100">
      <TopNav variant="app" />

      <div className="container-app py-8 space-y-6">
        {/* ------------------------------------------------------------------ */}
        {/* 1. HEADER & PRODUCT SUBTEXT                                        */}
        {/* ------------------------------------------------------------------ */}
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 border-b border-ink-200/70 pb-5 dark:border-slate-800">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-ink-900 dark:text-white">
              Standards intelligence
            </h1>
            <p className="mt-1 text-sm text-ink-500 dark:text-slate-400">
              Explore applicable, related and historical standards.
            </p>
          </div>


          <div className="flex items-center gap-2 shrink-0">
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<Columns size={14} />}
              onClick={() => {
                if (selectedForCompare.length < 2) {
                  setSelectedForCompare(['std-10322', 'std-1944']);
                }
                setIsCompareModalOpen(true);
              }}
            >
              Compare Standards
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<FileText size={14} />}
              onClick={() => navigate({ name: 'analysis', analysisId: 'an-001', tab: 'standards' })}
            >
              Active Analysis Standards
            </Button>
          </div>
        </div>

        {/* ------------------------------------------------------------------ */}
        {/* 2. RECENTLY RELEVANT / RECENTLY UPDATED STANDARDS STRIP            */}
        {/* ------------------------------------------------------------------ */}
        <div className="rounded-xl border border-teal-200/80 bg-teal-50/20 p-4 shadow-soft">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Sparkles size={16} className="text-teal-700" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-900 font-mono">
                Recently Relevant & Key Standards
              </h3>
            </div>
            <span className="text-[11px] text-teal-800 font-medium">
              Source-backed indexed intelligence
            </span>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            {featuredStandards.map((std) => (
              <div
                key={std.id}
                className="flex flex-col justify-between rounded-lg border border-ink-100 bg-white p-3 shadow-soft hover:border-teal-300 transition-all"
              >
                <div>
                  <div className="flex items-center justify-between gap-1 mb-1.5">
                    <span className="font-mono text-xs font-bold text-ink-900">{std.number}</span>
                    {renderStatusBadge(std.status)}
                  </div>
                  <p className="text-xs font-medium text-ink-800 line-clamp-1">{std.title}</p>
                  <p className="text-[11px] text-ink-500 mt-1 leading-relaxed line-clamp-2">
                    {std.whyApplies || std.summary}
                  </p>
                </div>

                <div className="mt-3 flex items-center justify-between border-t border-ink-100 pt-2 text-[11px]">
                  <span className="text-ink-400 font-mono">
                    Ed. {std.edition} ({std.revision})
                  </span>
                  <button
                    onClick={() => navigate({ name: 'standard', standardId: std.id })}
                    className="font-medium text-teal-700 hover:text-teal-900 inline-flex items-center gap-0.5"
                  >
                    View detail <ChevronRight size={12} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ------------------------------------------------------------------ */}
        {/* 3. SEARCH & ADVANCED FILTER CONTROLS                               */}
        {/* ------------------------------------------------------------------ */}
        <div className="space-y-3 rounded-xl border border-ink-200 bg-white p-4 shadow-soft">
          {/* Top Search Input */}
          <div className="relative">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-400" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by IS number, title, scope, product or description…"
              className="w-full rounded-lg border border-ink-200 bg-ivory-50/50 py-2 pl-9 pr-8 text-sm text-ink-900 placeholder:text-ink-400 focus:border-teal-500 focus:bg-white focus:outline-none"
            />
            {search && (
              <button
                onClick={() => setSearch('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-400 hover:text-ink-600"
              >
                <X size={15} />
              </button>
            )}
          </div>

          {/* Filter Rows: Status, Category, Relationship & Sorting */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-ink-100 text-xs">
            {/* Status Filters */}
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] font-semibold text-ink-400 uppercase font-mono mr-1">
                Status:
              </span>
              {[
                { id: 'all', label: 'All' },
                { id: 'current', label: 'CURRENT' },
                { id: 'under-review', label: 'NEEDS REVIEW' },
                { id: 'superseded', label: 'SUPERSEDED' },
                { id: 'withdrawn', label: 'WITHDRAWN' },
              ].map((f) => (
                <button
                  key={f.id}
                  onClick={() => setStatusFilter(f.id as any)}
                  className={`rounded px-2 py-1 text-[11px] font-mono font-medium transition-colors ${
                    statusFilter === f.id
                      ? 'bg-ink-900 text-white'
                      : 'bg-ivory-100 text-ink-600 hover:bg-ivory-200'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>

            {/* Category Filter & Sorting Dropdown */}
            <div className="flex flex-wrap items-center gap-2">
              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
                className="rounded border border-ink-200 bg-white px-2 py-1 text-xs text-ink-700 focus:border-teal-500 focus:outline-none max-w-xs"
              >
                <option value="all">All Categories</option>
                {Array.from(new Set(baseStandards.map(s => s.category).filter(Boolean))).sort().map((cat) => (
                  <option key={cat} value={cat!}>
                    {cat}
                  </option>
                ))}
              </select>

              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="rounded border border-ink-200 bg-white px-2 py-1 text-xs text-ink-700 focus:border-teal-500 focus:outline-none font-medium"
              >
                <option value="relevance">Sort: Relevance</option>
                <option value="current-first">Sort: Current First</option>
                <option value="recent-update">Sort: Recently Updated</option>
              </select>

              {(search || statusFilter !== 'all' || categoryFilter !== 'all' || relationshipFilter !== 'all') && (
                <button
                  onClick={() => {
                    setSearch('');
                    setStatusFilter('all');
                    setCategoryFilter('all');
                    setRelationshipFilter('all');
                  }}
                  className="text-[11px] text-teal-700 hover:underline ml-1"
                >
                  Reset filters
                </button>
              )}
            </div>
          </div>
        </div>

        {/* ------------------------------------------------------------------ */}
        {/* 4. STANDARDS RESULTS LIST                                          */}
        {/* ------------------------------------------------------------------ */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-ink-500 px-1 font-mono">
            <span>
              Showing {filteredStandards.length} indexed standards
            </span>
            <span>
              {selectedForCompare.length > 0 && `${selectedForCompare.length} selected for comparison`}
            </span>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            {filteredStandards.map((std, i) => {
              const isSelected = selectedForCompare.includes(std.id);
              const hasIssue = std.status !== 'current';

              return (
                <motion.div
                  key={std.id}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2, delay: Math.min(i * 0.02, 0.2) }}
                >
                  <Card
                    padding="md"
                    className={`bg-white transition-all flex flex-col justify-between h-full ${
                      isSelected
                        ? 'border-teal-500 ring-2 ring-teal-500/20'
                        : 'border-ink-200 hover:border-ink-300'
                    }`}
                  >
                    <div>
                      {/* Top standard header */}
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <div className="flex items-start gap-2.5">
                          <div
                            className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg font-mono text-xs font-bold ${
                              std.relationshipRole === 'primary'
                                ? 'bg-ink-900 text-teal-400'
                                : hasIssue
                                ? 'bg-warning-100 text-warning-800'
                                : 'bg-ivory-100 text-ink-700'
                            }`}
                          >
                            {hasIssue ? <FileWarning size={16} /> : <BookMarked size={16} />}
                          </div>
                          <div>
                            <div className="flex flex-wrap items-center gap-1.5">
                              <button
                                onClick={() => navigate({ name: 'standard', standardId: std.id })}
                                className="font-mono text-sm font-bold text-ink-900 hover:text-teal-700 flex items-center gap-1"
                              >
                                {std.number}
                                <ExternalLink size={11} className="text-ink-400" />
                              </button>
                              {renderRoleBadge(std.relationshipRole)}
                            </div>
                            <span className="text-[10px] text-ink-400 font-mono block">
                              {std.category || 'Engineering'} · Section {std.section}
                            </span>
                          </div>
                        </div>

                        {/* Status & Compare Checkbox */}
                        <div className="flex flex-col items-end gap-1.5 shrink-0">
                          {renderStatusBadge(std.status)}
                          <label className="flex items-center gap-1 text-[10px] text-ink-500 cursor-pointer select-none">
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => toggleCompare(std.id)}
                              className="rounded border-ink-300 text-teal-600 focus:ring-teal-500"
                            />
                            <span>Compare</span>
                          </label>
                        </div>
                      </div>

                      {/* Title & Summary */}
                      <h4 className="text-xs font-semibold text-ink-900 mb-1">{std.title}</h4>
                      <p className="text-[11px] text-ink-600 leading-relaxed line-clamp-2 mb-2">
                        {std.summary}
                      </p>

                      {/* Technical coverage snippet if available */}
                      {std.technicalCoverage && (
                        <p className="text-[10px] text-ink-500 bg-ivory-50 p-1.5 rounded border border-ink-100 line-clamp-1 font-mono mb-2">
                          <span className="font-semibold text-ink-700 font-sans">Coverage:</span> {std.technicalCoverage}
                        </p>
                      )}
                    </div>

                    {/* Footer metadata & buttons */}
                    <div className="border-t border-ink-100 pt-2 flex items-center justify-between text-[11px] text-ink-400 font-mono">
                      <div className="flex items-center gap-2">
                        <span>Ed. {std.edition}</span>
                        <span>·</span>
                        <span>{std.pages}p</span>
                        {std.evidenceAvailable && (
                          <>
                            <span>·</span>
                            <span className="text-teal-700 font-sans font-medium">
                              Evidence available
                            </span>
                          </>
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        {std.applicabilityScore !== undefined && (
                          <span className="font-semibold text-teal-800 bg-teal-50 px-1.5 py-0.5 rounded border border-teal-200">
                            {std.applicabilityScore}% match
                          </span>
                        )}
                        <button
                          onClick={() => navigate({ name: 'standard', standardId: std.id })}
                          className="font-sans font-medium text-teal-700 hover:text-teal-900 inline-flex items-center gap-0.5"
                        >
                          Detail <ChevronRight size={12} />
                        </button>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              );
            })}
          </div>

          {filteredStandards.length === 0 && (
            <Card padding="lg" className="text-center bg-white border-ink-200">
              <Filter size={24} className="mx-auto mb-2 text-ink-400" />
              <p className="text-sm font-semibold text-ink-900">No standards found</p>
              <p className="mt-1 text-xs text-ink-400">Try adjusting your keyword or status filters.</p>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  setSearch('');
                  setStatusFilter('all');
                  setCategoryFilter('all');
                  setRelationshipFilter('all');
                }}
                className="mt-4"
              >
                Clear all filters
              </Button>
            </Card>
          )}
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* 5. FLOATING COMPARISON TRAY                                        */}
      {/* ------------------------------------------------------------------ */}
      <AnimatePresence>
        {selectedForCompare.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 flex items-center gap-3 rounded-xl border border-ink-300 bg-white/95 px-5 py-3 shadow-2xl backdrop-blur-md"
          >
            <div className="flex items-center gap-2 text-xs">
              <Scale size={16} className="text-teal-700" />
              <span className="font-semibold text-ink-900">
                Comparing {selectedForCompare.length} standard{selectedForCompare.length > 1 ? 's' : ''}:
              </span>
              <div className="flex items-center gap-1.5 font-mono text-[11px] text-teal-800">
                {selectedForCompare.map((id) => {
                  const s = standards.find((item) => item.id === id);
                  return (
                    <span key={id} className="bg-teal-50 px-2 py-0.5 rounded border border-teal-200 font-semibold">
                      {s?.number || id}
                    </span>
                  );
                })}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                size="sm"
                onClick={() => setIsCompareModalOpen(true)}
                disabled={selectedForCompare.length < 2}
                leftIcon={<Columns size={14} />}
              >
                {selectedForCompare.length < 2 ? 'Select 1 more' : 'View Comparison'}
              </Button>
              <button
                onClick={() => setSelectedForCompare([])}
                className="rounded p-1 text-ink-400 hover:bg-ink-100 hover:text-ink-700"
                title="Clear comparison selection"
              >
                <X size={16} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ------------------------------------------------------------------ */}
      {/* 6. COMPARISON MATRIX MODAL                                         */}
      {/* ------------------------------------------------------------------ */}
      <StandardComparisonModal
        standardAId={selectedForCompare[0] || 'std-10322'}
        standardBId={selectedForCompare[1] || 'std-1944'}
        isOpen={isCompareModalOpen}
        onClose={() => setIsCompareModalOpen(false)}
        onSelectA={(id) => setSelectedForCompare((prev) => [id, prev[1] || 'std-1944'])}
        onSelectB={(id) => setSelectedForCompare((prev) => [prev[0] || 'std-10322', id])}
      />
    </div>
  );
}

