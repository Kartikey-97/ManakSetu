import type { ReactNode } from 'react';

type Variant = 'neutral' | 'teal' | 'blue' | 'success' | 'warning' | 'error' | 'outline';

interface BadgeProps {
  variant?: Variant;
  children: ReactNode;
  icon?: ReactNode;
  className?: string;
}

const variantMap: Record<Variant, string> = {
  neutral: 'badge-neutral',
  teal: 'badge-teal',
  blue: 'badge-blue',
  success: 'badge-success',
  warning: 'badge-warning',
  error: 'badge-error',
  outline: 'badge-outline',
};

export function Badge({ variant = 'neutral', icon, children, className = '' }: BadgeProps) {
  return (
    <span className={`${variantMap[variant]} ${className}`}>
      {icon}
      {children}
    </span>
  );
}
