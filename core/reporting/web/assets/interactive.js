/**
 * FirstLight Interactive Report JavaScript
 *
 * Provides interactivity for:
 * - Before/after image slider
 * - Collapsible sections
 * - Print functionality
 * - Mobile touch support
 */

(function() {
    'use strict';

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    function init() {
        initBeforeAfterSlider();
        initCollapsibleSections();
        initPrintButton();
        initKeyboardNav();
    }

    /**
     * Before/After Slider
     *
     * Allows dragging a slider to compare before/after images.
     * Supports mouse, touch, and keyboard input.
     */
    function initBeforeAfterSlider() {
        const sliders = document.querySelectorAll('.fl-before-after__slider');

        sliders.forEach(slider => {
            const container = slider.closest('.fl-before-after__wrapper');
            if (!container) return;

            const overlay = container.querySelector('.fl-before-after__overlay');
            if (!overlay) return;

            let isDragging = false;

            // Update slider position
            function updateSlider(x) {
                const rect = container.getBoundingClientRect();
                const offsetX = x - rect.left;
                const percentage = Math.max(0, Math.min(100, (offsetX / rect.width) * 100));
                overlay.style.width = percentage + '%';
            }

            // Mouse events
            slider.addEventListener('mousedown', (e) => {
                e.preventDefault();
                isDragging = true;
                document.body.style.cursor = 'ew-resize';
            });

            // Touch events
            slider.addEventListener('touchstart', (e) => {
                e.preventDefault();
                isDragging = true;
            }, { passive: false });

            // Move events
            document.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    e.preventDefault();
                    updateSlider(e.clientX);
                }
            });

            document.addEventListener('touchmove', (e) => {
                if (isDragging) {
                    e.preventDefault();
                    updateSlider(e.touches[0].clientX);
                }
            }, { passive: false });

            // End drag
            function endDrag() {
                isDragging = false;
                document.body.style.cursor = '';
            }

            document.addEventListener('mouseup', endDrag);
            document.addEventListener('touchend', endDrag);

            // Keyboard support (arrow keys to move slider)
            slider.setAttribute('tabindex', '0');
            slider.setAttribute('role', 'slider');
            slider.setAttribute('aria-label', 'Before/After comparison slider');
            slider.setAttribute('aria-valuemin', '0');
            slider.setAttribute('aria-valuemax', '100');
            slider.setAttribute('aria-valuenow', '50');

            slider.addEventListener('keydown', (e) => {
                const rect = container.getBoundingClientRect();
                const currentPercentage = (overlay.offsetWidth / container.offsetWidth) * 100;
                let newPercentage = currentPercentage;

                if (e.key === 'ArrowLeft' || e.key === 'Left') {
                    e.preventDefault();
                    newPercentage = Math.max(0, currentPercentage - 5);
                } else if (e.key === 'ArrowRight' || e.key === 'Right') {
                    e.preventDefault();
                    newPercentage = Math.min(100, currentPercentage + 5);
                }

                if (newPercentage !== currentPercentage) {
                    overlay.style.width = newPercentage + '%';
                    slider.setAttribute('aria-valuenow', Math.round(newPercentage));
                }
            });

            // Initialize at 50%
            const rect = container.getBoundingClientRect();
            updateSlider(rect.left + rect.width / 2);
        });
    }

    /**
     * Collapsible Sections
     *
     * Allows sections to be expanded/collapsed by clicking headers.
     * Supports keyboard navigation.
     */
    function initCollapsibleSections() {
        const headers = document.querySelectorAll('.fl-collapsible__header');

        headers.forEach((header, index) => {
            const collapsible = header.closest('.fl-collapsible');
            if (!collapsible) return;

            const content = collapsible.querySelector('.fl-collapsible__content');
            if (!content) return;

            // Make keyboard accessible
            header.setAttribute('tabindex', '0');
            header.setAttribute('role', 'button');
            header.setAttribute('aria-expanded', collapsible.classList.contains('fl-collapsible--expanded'));

            // Click handler
            header.addEventListener('click', () => {
                toggleCollapsible(collapsible, header);
            });

            // Keyboard handler
            header.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggleCollapsible(collapsible, header);
                }
            });
        });

        // Expand first section by default if none are expanded
        const firstCollapsible = document.querySelector('.fl-collapsible');
        if (firstCollapsible && !document.querySelector('.fl-collapsible--expanded')) {
            firstCollapsible.classList.add('fl-collapsible--expanded');
            const firstHeader = firstCollapsible.querySelector('.fl-collapsible__header');
            if (firstHeader) {
                firstHeader.setAttribute('aria-expanded', 'true');
            }
        }

        function toggleCollapsible(collapsible, header) {
            const isExpanded = collapsible.classList.toggle('fl-collapsible--expanded');
            header.setAttribute('aria-expanded', isExpanded);

            // Announce to screen readers
            const title = header.querySelector('.fl-collapsible__title');
            if (title) {
                const announcement = `${title.textContent} ${isExpanded ? 'expanded' : 'collapsed'}`;
                announceToScreenReader(announcement);
            }
        }
    }

    /**
     * Print Button
     *
     * Triggers browser print dialog.
     */
    function initPrintButton() {
        const printButtons = document.querySelectorAll('[data-action="print"]');

        printButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Expand all sections before printing
                const collapsibles = document.querySelectorAll('.fl-collapsible');
                collapsibles.forEach(c => c.classList.add('fl-collapsible--expanded'));

                // Trigger print
                window.print();
            });
        });
    }

    /**
     * Keyboard Navigation
     *
     * Enhances keyboard accessibility.
     */
    function initKeyboardNav() {
        // Add focus styles
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-nav');
            }
        });

        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-nav');
        });

        // Add focus styles CSS if not already present
        if (!document.getElementById('keyboard-nav-styles')) {
            const style = document.createElement('style');
            style.id = 'keyboard-nav-styles';
            style.textContent = `
                body.keyboard-nav *:focus {
                    outline: 2px solid #3182CE;
                    outline-offset: 2px;
                }
                body:not(.keyboard-nav) *:focus {
                    outline: none;
                }
            `;
            document.head.appendChild(style);
        }
    }

    /**
     * Utility: Announce to screen readers
     *
     * Creates a live region to announce dynamic changes.
     */
    function announceToScreenReader(message) {
        let announcer = document.getElementById('fl-announcer');

        if (!announcer) {
            announcer = document.createElement('div');
            announcer.id = 'fl-announcer';
            announcer.setAttribute('aria-live', 'polite');
            announcer.setAttribute('aria-atomic', 'true');
            announcer.style.position = 'absolute';
            announcer.style.left = '-10000px';
            announcer.style.width = '1px';
            announcer.style.height = '1px';
            announcer.style.overflow = 'hidden';
            document.body.appendChild(announcer);
        }

        announcer.textContent = message;

        // Clear after announcement
        setTimeout(() => {
            announcer.textContent = '';
        }, 1000);
    }

})();
