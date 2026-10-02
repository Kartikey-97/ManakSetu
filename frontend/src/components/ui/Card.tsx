import type { HTMLAttributes } from 'react';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  interactive?: boolean;
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

const paddingMap = {
  none: '',
  sm: 'p-4',
  md: 'p-5',
  lg: 'p-6 sm:p-8',
};

export function Card({ interactive, padding = 'md', className = '', children, ...props }: CardProps) {
  return (
    <div
      className={`${interactive ? 'card-interactive' : 'card'} ${paddingMap[padding]} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
