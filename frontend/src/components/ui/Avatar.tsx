interface AvatarProps {
  initials: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeMap = {
  sm: 'w-7 h-7 text-xs',
  md: 'w-9 h-9 text-sm',
  lg: 'w-11 h-11 text-base',
};

export function Avatar({ initials, size = 'md', className = '' }: AvatarProps) {
  return (
    <div
      className={`${sizeMap[size]} flex shrink-0 items-center justify-center rounded-full bg-ink-100 font-medium text-ink-700 ring-1 ring-inset ring-ink-200 ${className}`}
    >
      {initials}
    </div>
  );
}
