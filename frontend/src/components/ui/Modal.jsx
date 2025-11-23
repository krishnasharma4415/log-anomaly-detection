import { AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import { useEffect } from 'react';

export default function Modal({ isOpen, onClose, title, children, size = 'md' }) {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  const sizes = {
    sm: 'max-w-md',
    md: 'max-w-2xl',
    lg: 'max-w-4xl',
    xl: 'max-w-6xl',
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <div
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          />
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div
              className={`
                w-full ${sizes[size]} bg-neutral-surface rounded-xl shadow-card
                border border-neutral-border max-h-[90vh] overflow-hidden flex flex-col
              `}
            >
              <div className="flex items-center justify-between p-6 border-b border-neutral-border">
                <h2 className="text-xl font-semibold text-neutral-primary">{title}</h2>
                <button
                  onClick={onClose}
                  className="text-neutral-secondary hover:text-neutral-primary transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-6">
                {children}
              </div>
            </div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}
