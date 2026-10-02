import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';

type Variant = 'primary' | 'secondary' | 'ghost' | 'outline';
type Size = 'sm' | 'md' | 'lg';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  fullWidth?: boolean;
}

const sizeMap: Record<Size, string> = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2 text-sm',
  lg: 'px-5 py-2.5 text-sm',
};

const variantMap: Record<Variant, string> = {
  primary: 'btn bg-ink-900 text-white shadow-soft hover:bg-ink-800 focus:ring-ink-400 dark:bg-teal-700 dark:text-white dark:hover:bg-teal-600',
  secondary: 'btn border border-ink-200 bg-white text-ink-700 shadow-soft hover:border-ink-300 hover:bg-ivory-50 focus:ring-ink-300 dark:border-slate-700 dark:bg-[#161F30] dark:text-slate-200 dark:hover:border-slate-600 dark:hover:bg-slate-800',
  ghost: 'btn text-ink-600 hover:bg-ink-100 hover:text-ink-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white',
  outline: 'btn border border-ink-300 bg-transparent text-ink-700 hover:bg-ivory-50 focus:ring-ink-300 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800/80',
};


export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { variant = 'primary', size = 'md', leftIcon, rightIcon, fullWidth, className = '', children, ...props },
    ref
  ) => {
    return (
      <button
        ref={ref}
        className={`${variantMap[variant]} ${sizeMap[size]} ${fullWidth ? 'w-full' : ''} ${className}`}
        {...props}
      >
        {leftIcon}
        {children}
        {rightIcon}
      </button>
    );
  }
);
Button.displayName = 'Button';
