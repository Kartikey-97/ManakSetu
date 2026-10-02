import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Building2,
  Calendar,
  Check,
  CheckCircle2,
  ChevronRight,
  Clock,
  Columns,
  ExternalLink,
  FileCheck2,
  FileText,
  FileWarning,
  GitBranch,
  Globe,
  Hash,
  HelpCircle,
  History,
  Layers,
  Scale,
  ScrollText,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Users,
  X,
} from 'lucide-react';
import { TopNav } from '@/components/TopNav';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { useRouter } from '@/router';
import { adaptStandard } from '@/services/adapter';
import { getStandard } from '@/services/api';
import {
  standards,
  statusConfig,
  getStandardById,
  getMatchedRequirementsByAnalysisId,
  getEvidenceChainsByAnalysisId,
} from '@/data/mockData';
import type {
  HumanDecision,
  MatchedRequirementStatus,
  Standard,
  StandardRelationshipRole,
} from '@/data/types';
import { StandardComparisonModal } from '@/components/standards/StandardComparisonModal';

interface Props {
  standardId: string;
}

export function StandardDetailPage({ standardId }: Props) {
  const { route, navigate } = useRouter();
  const mockStandard = getStandardById(standardId);

  // For catalog standards not in the seeded/registered set, fetch the real record from the backend.
  const [fetchedStandard, setFetchedStandard] = useState<Standard | undefined>(undefined);
  const [loadingStandard, setLoadingStandard] = useState(false);

  useEffect(() => {
    if (mockStandard) return;
    let alive = true;
    setLoadingStandard(true);
    getStandard(standardId)
      .then((raw) => {
        if (alive && raw) setFetchedStandard(adaptStandard(raw));
      })
      .catch(() => {})
      .finally(() => {
        if (alive) setLoadingStandard(false);
      });
    return () => {
      alive = false;
    };
  }, [standardId, mockStandard]);

  const standard = mockStandard || fetchedStandard;

  if (loadingStandard) {
    return (
      <div className="min-h-screen bg-sand-50/50 flex items-center justify-center">
        <div className="text-ink-500 font-mono text-sm animate-pulse">Loading standard details...</div>
      </div>
    );
  }

  if (!standard) {
    return (
      <div className="min-h-screen bg-sand-50/50 flex flex-col items-center justify-center p-8">
        <AlertTriangle size={32} className="text-danger-500 mb-4" />
        <h2 className="text-xl font-bold text-ink-900 mb-2">Standard Not Found</h2>
        <p className="text-ink-600 mb-6 text-center max-w-md">The standard "{standardId}" could not be found or failed to load.</p>
        <Button variant="primary" onClick={() => window.history.back()}>Go Back</Button>
      </div>
    );
  }

  // Comparison modal state
  const [isCompareOpen, setIsCompareOpen] = useState(false);
  const [compareTargetId, setCompareTargetId] = useState<string>(
    standard?.supersededBy || standard?.references[0] || 'std-1944'
  );

  if (!standard) {
    return (
      <div className="min-h-screen bg-ivory-50">
        <TopNav variant="app" />
        <div className="container-app py-20 text-center">
          <p className="text-sm text-ink-500">
            {loadingStandard ? 'Loading standard from BIS catalog…' : 'Standard not found in indexed catalog.'}
          </p>
          {!loadingStandard && (
            <Button variant="secondary" onClick={() => navigate({ name: 'standards' })} className="mt-4">
              Back to Standards Intelligence
            </Button>
          )}
        </div>
      </div>
    );
  }

  const status = statusConfig[standard.status];
  const referencedBy = standard.referencedBy.map((id) => getStandardById(id)).filter((s): s is Standard => s !== undefined);
  const references = standard.references.map((id) => getStandardById(id)).filter((s): s is Standard => s !== undefined);
  const supersededBy = standard.supersededBy ? getStandardById(standard.supersededBy) : null;

  // Get the active analysis ID from the route, fallback to mock if none
  const activeAnalysisId = (route.name === 'standard' && route.analysisId) 
    ? route.analysisId 
    : 'an-001';

  // Filter matched requirements relevant to this standard
  const allRequirements = getMatchedRequirementsByAnalysisId(activeAnalysisId);
  const relevantRequirements = allRequirements.filter(
    (req) => req.standardId === standard.id || 
             req.standardIds?.includes(standard.id) || 
             req.standardCode.includes(standard.number.split(' ')[1] || '')
  );

  // Evidence chains linked to this standard
  const allEvidence = getEvidenceChainsByAnalysisId(activeAnalysisId);
  const relevantEvidence = allEvidence.filter((ev) => ev.standard.includes(standard.number));

  // Organize related references into categories
  const normativeRefs = references.filter((r) => r.relationshipRole === 'normative' || !r.relationshipRole);
  const testingRefs = references.filter((r) => r.relationshipRole === 'testing');
  const safetyRefs = references.filter((r) => r.relationshipRole === 'safety');
  const installationRefs = references.filter((r) => r.relationshipRole === 'installation');

  const renderReqStatusBadge = (status: MatchedRequirementStatus) => {
    switch (status) {
      case 'covered':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-success-50 px-2 py-0.5 text-[11px] font-medium text-success-800 border border-success-200/60 font-mono">
            <CheckCircle2 size={11} className="text-success-600" />
            Covered
          </span>
        );
      case 'partial':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-warning-50 px-2 py-0.5 text-[11px] font-medium text-warning-800 border border-warning-200/60 font-mono">
            <AlertTriangle size={11} className="text-warning-600" />
            Partial
          </span>
        );
      case 'needs-review':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-blue-50 px-2 py-0.5 text-[11px] font-medium text-blue-800 border border-blue-200/60 font-mono">
            <HelpCircle size={11} className="text-blue-600" />
            Needs review
          </span>
        );
      case 'not-found':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-ink-100 px-2 py-0.5 text-[11px] font-medium text-ink-700 border border-ink-200 font-mono">
            <X size={11} className="text-ink-500" />
            Not found
          </span>
        );
    }
  };

  const renderRoleBadge = (role?: StandardRelationshipRole) => {
    switch (role) {
      case 'primary':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-teal-800 text-white px-2 py-0.5 text-xs font-medium font-mono">
            Primary Applicable Code
          </span>
        );
      case 'normative':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-blue-50 text-blue-800 border border-blue-200 px-2 py-0.5 text-xs font-medium font-mono">
            Normative Reference
          </span>
        );
      case 'testing':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-purple-50 text-purple-800 border border-purple-200 px-2 py-0.5 text-xs font-medium font-mono">
            Testing Protocol
          </span>
        );
      case 'safety':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-amber-50 text-amber-900 border border-amber-200 px-2 py-0.5 text-xs font-medium font-mono">
            Safety Standard
          </span>
        );
      case 'installation':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 px-2 py-0.5 text-xs font-medium font-mono">
            Design & Installation
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 rounded bg-ink-100 text-ink-700 border border-ink-200 px-2 py-0.5 text-xs font-medium font-mono">
            Related Standard
          </span>
        );
    }
  };

  return (
    <div className="min-h-screen bg-ivory-50 text-ink-900 pb-16 dark:bg-[#090D16] dark:text-slate-100">
      <TopNav variant="app" />

      <div className="container-app py-6 space-y-6">
        {/* Breadcrumb Navigation */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-xs text-ink-500">
            <button
              onClick={() => navigate({ name: 'standards' })}
              className="flex items-center gap-1 hover:text-ink-900 transition-colors"
            >
              <ArrowLeft size={14} />
              Standards Intelligence
            </button>
            {route.name === 'standard' && route.analysisId && (
              <>
                <span>/</span>
                <button
                  onClick={() => navigate({ name: 'analysis', analysisId: route.analysisId!, tab: 'standards' })}
                  className="hover:text-ink-900 transition-colors"
                >
                  Return to Analysis
                </button>
              </>
            )}
            <span>/</span>
            <span className="font-mono text-ink-800 font-semibold">{standard.number}</span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<Columns size={14} />}
              onClick={() => {
                setCompareTargetId(standard.supersededBy || standard.references[0] || 'std-1944');
                setIsCompareOpen(true);
              }}
            >
              Compare with Another Standard
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<FileText size={14} />}
              onClick={() => navigate({ name: 'analysis', analysisId: activeAnalysisId, tab: 'standards' })}
            >
              Return to Analysis
            </Button>
          </div>
        </div>

        {/* ------------------------------------------------------------------ */}
        {/* TOP RESEARCH HEADER WITH PROMINENT STATUS BLOCK                    */}
        {/* ------------------------------------------------------------------ */}
        <Card padding="lg" className="bg-white border-ink-200 shadow-soft">
          <div className="flex flex-col md:flex-row md:items-start justify-between gap-5">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-ink-900 text-teal-400 font-mono font-bold text-sm shadow-soft">
                IS
              </div>
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h1 className="text-xl font-bold tracking-tight text-ink-900 sm:text-2xl font-mono">
                    {standard.number}
                  </h1>
                  {renderRoleBadge(standard.relationshipRole)}
                  <Badge variant={status.variant}>{status.label}</Badge>
                  {standard.isCertified && (
                    <Badge variant="teal" icon={<CheckCircle2 size={11} />}>
                      BIS ISI Marked
                    </Badge>
                  )}
                  {standard.regulatory && (
                    <Badge variant="blue" icon={<ShieldCheck size={11} />}>
                      Mandatory Quality Control Order
                    </Badge>
                  )}
                </div>
                <h2 className="mt-1.5 text-base font-semibold text-ink-800">{standard.title}</h2>
                <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-500 font-mono">
                  <span>Category: <strong>{standard.category || 'Electrical Equipment'}</strong></span>
                  <span>·</span>
                  <span>Edition {standard.edition} ({standard.revision})</span>
                  <span>·</span>
                  <span>Bureau: {standard.bureau} ({standard.section})</span>
                  <span>·</span>
                  <span>{standard.pages > 0 ? `${standard.pages} pages` : 'Metadata only'}</span>
                  <span>·</span>
                  <span>Published {standard.yearPublished}</span>
                </div>
              </div>
            </div>

            {/* Prominent Status & Applicability Block */}
            <div className="flex sm:flex-col items-center sm:items-end justify-between gap-2 shrink-0 border-t sm:border-t-0 pt-3 sm:pt-0 border-ink-100">
              <div className="rounded-xl border border-teal-200 bg-teal-50/50 p-3 text-right">
                <span className="block text-[10px] font-semibold uppercase tracking-wider text-teal-800 font-mono">
                  Current Status
                </span>
                <span className="font-mono text-xl font-bold text-teal-900">
                  {status.label.toUpperCase()}
                </span>
                <span className="block text-[11px] text-teal-700 mt-0.5">
                  Edition: {standard.edition}
                </span>
              </div>

              {standard.applicabilityScore !== undefined && (
                <span className="text-[11px] font-mono text-ink-500">
                  Retrieval match (last analysis): <strong className="text-ink-900 font-bold">{standard.applicabilityScore}%</strong>
                </span>
              )}
            </div>
          </div>
        </Card>

        {/* ------------------------------------------------------------------ */}
        {/* MAIN 2-COLUMN RESEARCH GRID                                        */}
        {/* ------------------------------------------------------------------ */}
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-6">
            {/* 1. OVERVIEW & SCOPE SECTION */}
            <Card padding="lg" className="bg-white border-ink-200 shadow-soft">
              <div className="flex items-center gap-2 mb-3 border-b border-ink-100 pb-2">
                <BookOpen size={16} className="text-teal-700" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-900 font-mono">
                  1. Scope & Primary Purpose
                </h3>
              </div>
              <p className="text-sm leading-relaxed text-ink-700">{standard.summary}</p>

              {standard.technicalCoverage && (
                <div className="mt-3 rounded-lg border border-ink-100 bg-ivory-50/60 p-3 text-xs">
                  <p className="font-semibold text-ink-800 text-[11px] uppercase tracking-wider mb-1">
                    Technical Parameter Coverage
                  </p>
                  <p className="text-ink-600 font-mono text-[11px] leading-relaxed">
                    {standard.technicalCoverage}
                  </p>
                </div>
              )}

              {standard.regulatoryNote && (
                <div className="mt-3 flex items-start gap-3 rounded-lg border border-blue-200 bg-blue-50/60 p-3.5">
                  <ShieldCheck size={18} className="mt-0.5 shrink-0 text-blue-600" />
                  <div className="text-xs">
                    <p className="font-semibold uppercase tracking-wide text-blue-900">
                      Statutory Regulatory Mandate
                    </p>
                    <p className="mt-0.5 text-ink-700 leading-relaxed">{standard.regulatoryNote}</p>
                  </div>
                </div>
              )}

              {supersededBy && (
                <div className="mt-3 flex items-start gap-3 rounded-lg border border-error-200 bg-error-50/60 p-3.5">
                  <FileWarning size={18} className="mt-0.5 shrink-0 text-error-600" />
                  <div className="text-xs">
                    <p className="font-semibold uppercase tracking-wide text-error-900">
                      Standard Superseded / Withdrawn
                    </p>
                    <p className="mt-0.5 text-ink-700 leading-relaxed">
                      This edition has been superseded by{' '}
                      <button
                        onClick={() => navigate({ name: 'standard', standardId: supersededBy.id })}
                        className="font-semibold text-teal-800 hover:underline inline-flex items-center gap-0.5"
                      >
                        {supersededBy.number} <ArrowRight size={11} />
                      </button>
                      . Ensure specifications cite the current standard to avoid audit objections.
                    </p>
                  </div>
                </div>
              )}
            </Card>

            {/* 2. DEDICATED VERSION INTELLIGENCE SECTION (Visual Chronology) */}
            <Card padding="lg" className="bg-white border-ink-200 shadow-soft">
              <div className="flex items-center justify-between mb-4 border-b border-ink-100 pb-2">
                <div className="flex items-center gap-2">
                  <History size={16} className="text-teal-700" />
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-900 font-mono">
                    2. Version Intelligence & Chronology
                  </h3>
                </div>
                <span className="text-xs text-ink-400 font-mono">
                  Indexed BIS Life Cycle
                </span>
              </div>

              {/* Chronology visual sequence: Previous -> Amended -> Current */}
              <div className="relative border-l-2 border-teal-200 ml-4 pl-6 space-y-6 text-xs">
                {/* 1. Previous / Origin Code */}
                <div className="relative">
                  <div className="absolute -left-[31px] top-0 flex h-5 w-5 items-center justify-center rounded-full bg-ink-200 text-ink-600 text-[10px] font-mono font-bold">
                    1
                  </div>
                  <div>
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-400 font-mono block">
                      PREVIOUS / HISTORICAL CODE
                    </span>
                    <span className="font-mono text-xs font-semibold text-ink-800">
                      {standard.previousEdition || 'Origin code not indexed'}
                    </span>
                    <p className="text-[11px] text-ink-500 mt-0.5 leading-relaxed">
                      {standard.previousEdition
                        ? 'Previous governing edition superseded upon formal BIS publication of current standard.'
                        : 'Historical superseded versions are not currently available in the intelligence index.'}
                    </p>
                  </div>
                </div>

                {/* 2. Published Amendments */}
                <div className="relative">
                  <div className="absolute -left-[31px] top-0 flex h-5 w-5 items-center justify-center rounded-full bg-teal-100 text-teal-800 text-[10px] font-mono font-bold">
                    2
                  </div>
                  <div>
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-teal-800 font-mono block">
                      AMENDMENTS INCORPORATED
                    </span>
                    {standard.amendments && standard.amendments.length > 0 ? (
                      <div className="mt-1 space-y-1">
                        {standard.amendments.map((am: any, idx) => (
                          <div key={idx} className="flex items-center gap-1.5 font-mono text-[11px] text-ink-700 bg-ivory-50 p-1.5 rounded border border-ink-100">
                            <Check size={11} className="text-teal-600 shrink-0" />
                            <span>{typeof am === 'string' ? am : `Amd. ${am.amendment_number || (idx+1)}${am.year ? ` (${am.year})` : ''}`}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <span className="text-[11px] text-ink-500 italic">
                        Not available in indexed data / No separate amendments
                      </span>
                    )}
                  </div>
                </div>

                {/* 3. Current Applicable Edition */}
                <div className="relative">
                  <div className="absolute -left-[31px] top-0 flex h-5 w-5 items-center justify-center rounded-full bg-teal-700 text-white text-[10px] font-mono font-bold shadow-soft">
                    3
                  </div>
                  <div className="rounded-lg border border-teal-200 bg-teal-50/40 p-3">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-teal-900 font-mono block">
                      CURRENT APPLICABLE EDITION
                    </span>
                    <span className="font-mono text-sm font-bold text-teal-950 block mt-0.5">
                      Edition {standard.edition} ({standard.revision})
                    </span>
                    <p className="text-[11px] text-ink-600 mt-1 leading-relaxed">
                      Confirmed current active standard with valid reaffirmation cycle. Applicable for ongoing public tenders and technical evaluation.
                    </p>
                  </div>
                </div>
              </div>
            </Card>

            {/* 3. WHY THIS STANDARD IS RELEVANT */}
            <Card padding="lg" className="bg-white border-ink-200 shadow-soft">
              <div className="flex items-center justify-between mb-3 border-b border-ink-100 pb-2">
                <div className="flex items-center gap-2">
                  <Scale size={16} className="text-teal-700" />
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-900 font-mono">
                    3. Why this standard is relevant
                  </h3>
                </div>
                <Badge variant="teal" className="text-[10px]">Active Procurement Match</Badge>
              </div>

              <p className="text-sm text-ink-700 leading-relaxed mb-3">
                {standard.whyApplies ||
                  'This standard applies directly to the active procurement profile based on equipment classification and environmental requirements.'}
              </p>

              {standard.whyAppliesReasons && (
                <div className="grid gap-2 sm:grid-cols-2 mb-4">
                  {standard.whyAppliesReasons.map((r, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-2 rounded-lg border border-ink-100 bg-ivory-50/60 p-2.5 text-xs"
                    >
                      <CheckCircle2 size={14} className="mt-0.5 shrink-0 text-teal-600" />
                      <div>
                        <span className="font-semibold text-ink-900 text-[11px]">{r.category}: </span>
                        <span className="text-ink-600 text-[11px]">{r.description}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Matched Requirements Subsection */}
              <div className="border-t border-ink-100 pt-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-ink-900 text-xs flex items-center gap-1.5">
                    <FileCheck2 size={14} className="text-teal-700" />
                    Matched Technical Requirements ({relevantRequirements.length})
                  </span>
                  <span className="text-[11px] text-ink-400 font-mono">Clause Mapping</span>
                </div>

                {relevantRequirements.length > 0 ? (
                  <div className="space-y-2">
                    {relevantRequirements.map((req) => (
                      <details
                        key={req.id}
                        className="group rounded-lg border border-ink-100 bg-ivory-50/40 text-xs"
                      >
                        <summary className="cursor-pointer p-2.5 flex items-center justify-between gap-2 list-none">
                          <div className="flex items-start gap-2">
                            <span className="text-ink-400 mt-0.5">•</span>
                            <span className="text-ink-800 line-clamp-2">{req.requirement}</span>
                          </div>
                          {renderReqStatusBadge(req.status)}
                        </summary>
                        
                        <div className="p-3 pt-0 border-t border-ink-100 bg-white/50 mt-1 space-y-2">
                          <div className="text-ink-600">
                            <span className="font-semibold">Tender Location: </span>
                            <span className="text-ink-900">{req.clause || 'Unspecified'}</span>
                          </div>
                          {req.evidenceSnippet && (
                            <div className="bg-ink-50 p-2 rounded border border-ink-100 text-ink-600">
                              <span className="font-semibold">Standard Proof: </span>
                              <span className="text-ink-900 italic">"{req.evidenceSnippet}"</span>
                            </div>
                          )}
                        </div>
                      </details>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-ink-400 italic">
                    Not directly mapped to active tender numerical parameters.
                  </p>
                )}
              </div>
            </Card>

            {/* 4. RELATED & NORMATIVE REFERENCES */}
            <Card padding="lg" className="bg-white border-ink-200 shadow-soft">
              <div className="flex items-center justify-between mb-3 border-b border-ink-100 pb-2">
                <div className="flex items-center gap-2">
                  <GitBranch size={16} className="text-teal-700" />
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-900 font-mono">
                    4. Related & Normative References
                  </h3>
                </div>
                <button
                  onClick={() => navigate({ name: 'analysis', analysisId: activeAnalysisId, tab: 'relationships' })}
                  className="text-xs text-teal-700 hover:text-teal-900 font-medium inline-flex items-center gap-0.5"
                >
                  Explore relationships <ArrowRight size={11} />
                </button>
              </div>

              <div className="space-y-4 text-xs">
                {/* Normative References */}
                {normativeRefs.length > 0 && (
                  <div>
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-500 font-mono block mb-1.5">
                      Normative References ({normativeRefs.length})
                    </span>
                    <div className="space-y-1.5">
                      {normativeRefs.map((ref) => (
                        <div
                          key={ref.id}
                          className="flex items-center justify-between rounded-lg border border-ink-100 bg-white p-2 text-left"
                        >
                          <div className="min-w-0 flex-1">
                            <button
                              onClick={() => navigate({ name: 'standard', standardId: ref.id })}
                              className="font-mono font-semibold text-ink-900 hover:text-teal-700"
                            >
                              {ref.number}
                            </button>
                            <span className="text-ink-500 text-[11px] truncate block">{ref.title}</span>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            {renderRoleBadge(ref.relationshipRole)}
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setCompareTargetId(ref.id);
                                setIsCompareOpen(true);
                              }}
                            >
                              Compare
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Testing & Safety references */}
                {(testingRefs.length > 0 || safetyRefs.length > 0 || installationRefs.length > 0) && (
                  <div>
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-500 font-mono block mb-1.5">
                      Testing, Safety & Design Companions
                    </span>
                    <div className="space-y-1.5">
                      {[...testingRefs, ...safetyRefs, ...installationRefs].map((ref) => (
                        <div
                          key={ref.id}
                          className="flex items-center justify-between rounded-lg border border-ink-100 bg-white p-2 text-left"
                        >
                          <div className="min-w-0 flex-1">
                            <button
                              onClick={() => navigate({ name: 'standard', standardId: ref.id })}
                              className="font-mono font-semibold text-ink-900 hover:text-teal-700"
                            >
                              {ref.number}
                            </button>
                            <span className="text-ink-500 text-[11px] truncate block">{ref.title}</span>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            {renderRoleBadge(ref.relationshipRole)}
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setCompareTargetId(ref.id);
                                setIsCompareOpen(true);
                              }}
                            >
                              Compare
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* International Equivalents */}
                {standard.internationalEquivalents && standard.internationalEquivalents.length > 0 && (
                  <div>
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-500 font-mono block mb-1.5">
                      International / Equivalent Standards
                    </span>
                    <div className="flex flex-wrap gap-1.5 font-mono text-[11px]">
                      {standard.internationalEquivalents.map((ie, idx) => (
                        <span key={idx} className="bg-ivory-100 px-2 py-0.5 rounded border border-ink-200 text-ink-800 flex items-center gap-1">
                          <Globe size={11} className="text-ink-400" />
                          {ie}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </Card>

            {/* 5. SOURCE EVIDENCE PROVENANCE */}
            <Card padding="lg" className="bg-white border-ink-200 shadow-soft">
              <div className="flex items-center justify-between mb-3 border-b border-ink-100 pb-2">
                <div className="flex items-center gap-2">
                  <ScrollText size={16} className="text-teal-700" />
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-900 font-mono">
                    5. Source Evidence & Testing Verification
                  </h3>
                </div>
                <Badge variant="teal" className="text-[10px]">Evidence Available</Badge>
              </div>

              {relevantEvidence.length > 0 ? (
                <div className="space-y-3">
                  {relevantEvidence.map((ev) => (
                    <div key={ev.id} className="rounded-lg border border-ink-200 bg-ivory-50/60 p-3 text-xs">
                      <div className="flex items-center justify-between text-ink-500 font-mono text-[11px] mb-1.5">
                        <span>Source: {ev.sourceDoc}</span>
                        <span>{ev.sourceLocation}</span>
                      </div>
                      <blockquote className="border-l-2 border-teal-500 pl-3 text-xs italic text-ink-800 bg-white p-2 rounded-r font-mono mb-2">
                        {ev.evidence}
                      </blockquote>
                      <div className="grid gap-2 sm:grid-cols-2">
                        <div className="rounded bg-white p-2 border border-ink-100">
                          <span className="font-semibold text-ink-800 text-[10px] uppercase">Governing Clause</span>
                          <p className="text-xs font-mono text-teal-800 mt-0.5">{ev.clause}</p>
                        </div>
                        <div className="rounded bg-white p-2 border border-ink-100">
                          <span className="font-semibold text-ink-800 text-[10px] uppercase">Evaluation Conclusion</span>
                          <p className="text-xs text-ink-700 mt-0.5">{ev.conclusion}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-lg border border-ink-100 bg-ivory-50 p-3 text-xs text-ink-600">
                  <p className="font-medium text-ink-800">Clause citations available in master specification index.</p>
                  <p className="mt-0.5">Laboratory test certificates must conform to NABL accredited guidelines.</p>
                </div>
              )}
            </Card>
          </div>

          {/* SIDEBAR METADATA (1 Column) */}
          <div className="space-y-4">
            <Card padding="md" className="bg-white border-ink-200 shadow-soft">
              <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-ink-500 font-mono">
                Publication Metadata
              </h3>
              <dl className="space-y-2 text-xs">
                {[
                  { icon: <Hash size={13} />, label: 'Standard Code', value: standard.number },
                  { icon: <Calendar size={13} />, label: 'Current Edition', value: standard.edition },
                  { icon: <Layers size={13} />, label: 'Revision Cycle', value: standard.revision },
                  { icon: <Calendar size={13} />, label: 'Year Published', value: String(standard.yearPublished) },
                  { icon: <Clock size={13} />, label: 'Last Indexed Update', value: standard.lastUpdatedDate || '2022' },
                  { icon: <FileText size={13} />, label: 'Page Count', value: standard.pages > 0 ? `${standard.pages} pages` : 'Metadata only' },
                  { icon: <BookOpen size={13} />, label: 'Sectional Committee', value: standard.section },
                  { icon: <Building2 size={13} />, label: 'Standardization Body', value: standard.bureau },
                ].map((detail) => (
                  <div key={detail.label} className="flex items-center justify-between gap-2 border-b border-ink-100/60 pb-1.5">
                    <dt className="flex items-center gap-1.5 text-ink-400">
                      {detail.icon}
                      {detail.label}
                    </dt>
                    <dd className="font-mono font-medium text-ink-900 text-right">{detail.value}</dd>
                  </div>
                ))}
              </dl>
            </Card>

            {/* Quick Compare Action Box */}
            <Card padding="md" className="border-teal-200 bg-teal-50/40 shadow-soft">
              <div className="flex items-start gap-2.5">
                <Scale size={18} className="text-teal-700 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-semibold text-ink-900 uppercase tracking-wider font-mono">
                    Compare Technical Specifications
                  </h4>
                  <p className="text-[11px] text-ink-600 mt-1 leading-relaxed">
                    Compare scope, parameters, and testing criteria of {standard.number} side-by-side with companion or superseded standards.
                  </p>
                  <Button
                    size="sm"
                    className="mt-3 w-full"
                    onClick={() => {
                      setCompareTargetId(standard.supersededBy || standard.references[0] || 'std-1944');
                      setIsCompareOpen(true);
                    }}
                    leftIcon={<Columns size={14} />}
                  >
                    Open Comparison Matrix
                  </Button>
                </div>
              </div>
            </Card>

            {/* Indexed Keywords */}
            <Card padding="md" className="bg-white border-ink-200 shadow-soft">
              <h3 className="mb-2.5 text-xs font-semibold uppercase tracking-wider text-ink-500 font-mono">
                Indexed Keywords
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {standard.keywords.map((kw) => (
                  <span
                    key={kw}
                    className="rounded bg-ivory-100 px-2 py-0.5 text-[11px] font-medium text-ink-700 border border-ink-200"
                  >
                    {kw}
                  </span>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </div>

      {/* Comparison Modal */}
      <StandardComparisonModal
        standardAId={standard.id}
        standardBId={compareTargetId}
        isOpen={isCompareOpen}
        onClose={() => setIsCompareOpen(false)}
      />
    </div>
  );
}


