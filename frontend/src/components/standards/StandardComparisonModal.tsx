import { useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  BookMarked,
  CheckCircle2,
  Columns,
  ExternalLink,
  HelpCircle,
  Scale,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { useRouter } from '@/router';
import { standards, statusConfig, getStandardById } from '@/data/mockData';
import type { Standard } from '@/data/types';

interface Props {
  standardAId: string;
  standardBId: string;
  isOpen: boolean;
  onClose: () => void;
  onSelectA?: (id: string) => void;
  onSelectB?: (id: string) => void;
}

type MatchType = 'same' | 'different' | 'a-only' | 'b-only' | 'needs-review';

interface ComparisonRow {
  property: string;
  category: string;
  valA: string;
  valB: string;
  matchType: MatchType;
  note?: string;
}

export function StandardComparisonModal({
  standardAId,
  standardBId,
  isOpen,
  onClose,
  onSelectA,
  onSelectB,
}: Props) {
  const { navigate } = useRouter();
  const [selectedA, setSelectedA] = useState(standardAId);
  const [selectedB, setSelectedB] = useState(standardBId);

  if (!isOpen) return null;

  const stdA = getStandardById(selectedA) || standards[0];
  const stdB = getStandardById(selectedB) || standards[1] || standards[0];

  const statusA = statusConfig[stdA.status];
  const statusB = statusConfig[stdB.status];

  const handleSwap = () => {
    const temp = selectedA;
    setSelectedA(selectedB);
    setSelectedB(temp);
  };

  const renderMatchBadge = (type: MatchType) => {
    switch (type) {
      case 'same':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-teal-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-teal-800 border border-teal-200 font-mono">
            <CheckCircle2 size={10} /> Same
          </span>
        );
      case 'different':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-blue-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-blue-800 border border-blue-200 font-mono">
            Different
          </span>
        );
      case 'a-only':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-amber-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-amber-900 border border-amber-200 font-mono">
            Standard A Only
          </span>
        );
      case 'b-only':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-purple-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-purple-900 border border-purple-200 font-mono">
            Standard B Only
          </span>
        );
      case 'needs-review':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-warning-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-warning-800 border border-warning-200 font-mono">
            <AlertTriangle size={10} /> Needs Review
          </span>
        );
    }
  };

  // Build matrix rows
  const comparisonRows: ComparisonRow[] = [
    {
      category: 'Lifecycle & Validity',
      property: 'Current Edition & Status',
      valA: `${stdA.edition} (${stdA.revision}) — ${statusA.label.toUpperCase()}`,
      valB: `${stdB.edition} (${stdB.revision}) — ${statusB.label.toUpperCase()}`,
      matchType: stdA.status === stdB.status ? 'same' : 'different',
      note: stdA.status === 'withdrawn' || stdB.status === 'withdrawn' ? 'Withdrawn standard requires corrigendum.' : undefined,
    },
    {
      category: 'Lifecycle & Validity',
      property: 'Year Published / Last Update',
      valA: `Published ${stdA.yearPublished} (Indexed update: ${stdA.lastUpdatedDate || '2022'})`,
      valB: `Published ${stdB.yearPublished} (Indexed update: ${stdB.lastUpdatedDate || '2020'})`,
      matchType: stdA.yearPublished === stdB.yearPublished ? 'same' : 'different',
    },
    {
      category: 'Procurement Applicability',
      property: 'Role & Retrieval Match',
      valA: `${stdA.relationshipRole ? stdA.relationshipRole.toUpperCase() : 'STANDARD'} (${stdA.applicabilityScore || '—'}% match)`,
      valB: `${stdB.relationshipRole ? stdB.relationshipRole.toUpperCase() : 'STANDARD'} (${stdB.applicabilityScore || '—'}% match)`,
      matchType: 'different',
      note: 'Based on current active procurement profile analysis.',
    },
    {
      category: 'Scope & Technical Coverage',
      property: 'Product Scope & Application',
      valA: stdA.summary,
      valB: stdB.summary,
      matchType: 'different',
    },
    {
      category: 'Scope & Technical Coverage',
      property: 'Technical Parameter Coverage',
      valA: stdA.technicalCoverage || 'Enclosure design, electrical insulation, and performance limits.',
      valB: stdB.technicalCoverage || 'Component safety, operational tolerances, and material specifications.',
      matchType: stdA.technicalCoverage && stdB.technicalCoverage ? 'different' : 'needs-review',
    },
    {
      category: 'Verification & Testing',
      property: 'Laboratory & Testing Protocols',
      valA: stdA.testingRequirements || 'NABL accredited test report verification required.',
      valB: stdB.testingRequirements || 'Standard routine test report verification.',
      matchType: 'different',
    },
    {
      category: 'Statutory Mandates',
      property: 'Certification & Regulatory Orders',
      valA: stdA.isCertified
        ? `Mandatory (${stdA.certificationBody || 'BIS ISI / CRS'}) ${stdA.regulatory ? '— Quality Control Order' : ''}`
        : 'Voluntary standard / Reference code',
      valB: stdB.isCertified
        ? `Mandatory (${stdB.certificationBody || 'BIS ISI / CRS'}) ${stdB.regulatory ? '— Quality Control Order' : ''}`
        : 'Voluntary standard / Reference code',
      matchType: stdA.isCertified === stdB.isCertified ? 'same' : 'different',
    },
    {
      category: 'Version History',
      property: 'Previous Edition & Supersession',
      valA: stdA.previousEdition ? `Replaced ${stdA.previousEdition}` : 'First edition / Initial release',
      valB: stdB.previousEdition ? `Replaced ${stdB.previousEdition}` : 'First edition / Initial release',
      matchType: stdA.previousEdition && stdB.previousEdition ? 'different' : 'needs-review',
    },
    {
      category: 'Version History',
      property: 'Published Amendments',
      valA: stdA.amendments && stdA.amendments.length > 0 ? stdA.amendments.join('; ') : 'No published amendments',
      valB: stdB.amendments && stdB.amendments.length > 0 ? stdB.amendments.join('; ') : 'No published amendments',
      matchType: stdA.amendments?.length === stdB.amendments?.length ? 'same' : 'different',
    },
    {
      category: 'International Alignment',
      property: 'Equivalent International Codes',
      valA: stdA.internationalEquivalents && stdA.internationalEquivalents.length > 0
        ? stdA.internationalEquivalents.join(', ')
        : 'Not available in indexed data',
      valB: stdB.internationalEquivalents && stdB.internationalEquivalents.length > 0
        ? stdB.internationalEquivalents.join(', ')
        : 'Not available in indexed data',
      matchType: stdA.internationalEquivalents?.length ? 'different' : 'needs-review',
    },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink-900/60 p-4 backdrop-blur-xs animate-fade">
      <div className="flex max-h-[90vh] w-full max-w-5xl flex-col rounded-xl border border-ink-200 bg-white shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-ink-100 bg-ivory-50 px-6 py-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-ink-900 text-teal-400">
              <Scale size={18} />
            </div>
            <div>
              <h2 className="text-base font-semibold tracking-tight text-ink-900">
                Standards Comparison Matrix
              </h2>
              <p className="text-xs text-ink-500">
                Side-by-side technical, version, and regulatory comparison
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={handleSwap} leftIcon={<Columns size={14} />}>
              Swap Columns
            </Button>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-ink-400 hover:bg-ink-100 hover:text-ink-700 transition-colors"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Standard Selectors Bar */}
        <div className="grid grid-cols-2 gap-4 border-b border-ink-100 bg-white p-4">
          {/* Standard A Selector */}
          <div className="rounded-lg border border-teal-200 bg-teal-50/30 p-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-teal-800 font-mono">
                Standard A
              </span>
              <Badge variant={statusA.variant}>{statusA.label}</Badge>
            </div>
            <select
              value={selectedA}
              onChange={(e) => {
                setSelectedA(e.target.value);
                if (onSelectA) onSelectA(e.target.value);
              }}
              className="w-full rounded border border-ink-200 bg-white px-2.5 py-1.5 text-xs font-semibold text-ink-900 focus:border-teal-500 focus:outline-none"
            >
              {standards.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.number} — {s.title.substring(0, 45)}…
                </option>
              ))}
            </select>
            <div className="mt-2 flex items-center justify-between text-[11px] text-ink-600">
              <span>Edition: <strong className="font-mono">{stdA.edition}</strong></span>
              <button
                onClick={() => {
                  onClose();
                  navigate({ name: 'standard', standardId: stdA.id });
                }}
                className="text-teal-700 hover:underline inline-flex items-center gap-0.5"
              >
                View full detail <ExternalLink size={10} />
              </button>
            </div>
          </div>

          {/* Standard B Selector */}
          <div className="rounded-lg border border-blue-200 bg-blue-50/30 p-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-blue-800 font-mono">
                Standard B
              </span>
              <Badge variant={statusB.variant}>{statusB.label}</Badge>
            </div>
            <select
              value={selectedB}
              onChange={(e) => {
                setSelectedB(e.target.value);
                if (onSelectB) onSelectB(e.target.value);
              }}
              className="w-full rounded border border-ink-200 bg-white px-2.5 py-1.5 text-xs font-semibold text-ink-900 focus:border-blue-500 focus:outline-none"
            >
              {standards.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.number} — {s.title.substring(0, 45)}…
                </option>
              ))}
            </select>
            <div className="mt-2 flex items-center justify-between text-[11px] text-ink-600">
              <span>Edition: <strong className="font-mono">{stdB.edition}</strong></span>
              <button
                onClick={() => {
                  onClose();
                  navigate({ name: 'standard', standardId: stdB.id });
                }}
                className="text-blue-700 hover:underline inline-flex items-center gap-0.5"
              >
                View full detail <ExternalLink size={10} />
              </button>
            </div>
          </div>
        </div>

        {/* Comparison Matrix Table */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="space-y-4">
            {['Lifecycle & Validity', 'Procurement Applicability', 'Scope & Technical Coverage', 'Verification & Testing', 'Statutory Mandates', 'Version History', 'International Alignment'].map(
              (category) => {
                const rows = comparisonRows.filter((r) => r.category === category);
                if (rows.length === 0) return null;

                return (
                  <div key={category} className="rounded-lg border border-ink-100 bg-white shadow-soft overflow-hidden">
                    <div className="bg-ivory-100/70 px-4 py-2 border-b border-ink-100">
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-800 font-mono">
                        {category}
                      </h4>
                    </div>

                    <div className="divide-y divide-ink-100">
                      {/* Column Headers for Data */ }
                      <div className="grid grid-cols-12 gap-3 px-3 py-1.5 bg-ivory-50/80 border-b border-ink-100 text-[10px] font-semibold uppercase tracking-wider text-ink-500 font-mono">
                        <div className="col-span-12 sm:col-span-3">Property</div>
                        <div className="col-span-12 sm:col-span-4 text-teal-800 truncate">{stdA.number}</div>
                        <div className="col-span-12 sm:col-span-5 text-blue-800 truncate">{stdB.number}</div>
                      </div>
                      
                      {rows.map((row, idx) => (
                        <div key={idx} className="grid grid-cols-12 gap-3 p-3 text-xs hover:bg-ivory-50/50 transition-colors">
                          <div className="col-span-12 sm:col-span-3">
                            <span className="font-semibold text-ink-900 block">{row.property}</span>
                            <div className="mt-1">{renderMatchBadge(row.matchType)}</div>
                            {row.note && (
                              <p className="mt-1 text-[10px] text-warning-700 leading-tight italic">
                                {row.note}
                              </p>
                            )}
                          </div>

                          <div className="col-span-12 sm:col-span-4 rounded bg-teal-50/10 p-2 border border-teal-100/40 font-mono text-[11px] text-ink-800 leading-relaxed">
                            {row.valA}
                          </div>

                          <div className="col-span-12 sm:col-span-5 rounded bg-blue-50/10 p-2 border border-blue-100/40 font-mono text-[11px] text-ink-800 leading-relaxed">
                            {row.valB}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              }
            )}
          </div>
        </div>

        {/* Footer note & action */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-ink-100 bg-ivory-50 px-6 py-3 text-xs text-ink-500">
          <span className="flex items-center gap-1.5 text-[11px]">
            <CheckCircle2 size={13} className="text-teal-600" />
            Comparison generated from indexed BIS catalog and analysis parameter mappings.
          </span>
          <Button variant="secondary" size="sm" onClick={onClose}>
            Close Comparison
          </Button>
        </div>
      </div>
    </div>
  );
}
