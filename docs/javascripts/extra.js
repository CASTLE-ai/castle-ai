// CASTLE Documentation JavaScript Enhancements

(function() {
    'use strict';

    // ==================
    // Page Load Handler
    // ==================
    document.addEventListener('DOMContentLoaded', function() {
        initializeFeatures();
        setupProgressTracking();
        setupVideoEnhancements();
        setupSearchEnhancements();
        setupAccessibilityFeatures();
    });

    // ==================
    // Feature Cards Animation
    // ==================
    function initializeFeatures() {
        // Animate feature cards on scroll
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    entry.target.style.transform = 'translateY(0)';
                    entry.target.style.opacity = '1';
                } else {
                    entry.target.style.transform = 'translateY(20px)';
                    entry.target.style.opacity = '0';
                }
            });
        }, observerOptions);

        // Observe all feature cards
        const featureCards = document.querySelectorAll('.feature-card');
        featureCards.forEach(function(card) {
            card.style.transform = 'translateY(20px)';
            card.style.opacity = '0';
            card.style.transition = 'all 0.6s ease-out';
            observer.observe(card);
        });

        // Add click handlers for interactive elements
        setupInteractiveElements();
    }

    // ==================
    // Progress Tracking
    // ==================
    function setupProgressTracking() {
        // Track checkbox interactions in learning paths
        const checkboxes = document.querySelectorAll('.learning-path input[type="checkbox"]');
        
        checkboxes.forEach(function(checkbox) {
            checkbox.addEventListener('change', function() {
                const pathSection = this.closest('.learning-path');
                if (pathSection) {
                    updatePathProgress(pathSection);
                }
                saveProgress();
            });
        });

        // Load saved progress
        loadProgress();
    }

    function updatePathProgress(pathSection) {
        const checkboxes = pathSection.querySelectorAll('input[type="checkbox"]');
        const checkedCount = pathSection.querySelectorAll('input[type="checkbox"]:checked').length;
        const progress = (checkedCount / checkboxes.length) * 100;

        // Create or update progress bar
        let progressBar = pathSection.querySelector('.progress-bar');
        if (!progressBar) {
            progressBar = document.createElement('div');
            progressBar.className = 'progress-bar';
            progressBar.innerHTML = `
                <div class="progress-track">
                    <div class="progress-fill"></div>
                </div>
                <span class="progress-text">進度: 0%</span>
            `;
            pathSection.insertBefore(progressBar, pathSection.firstChild);
        }

        const progressFill = progressBar.querySelector('.progress-fill');
        const progressText = progressBar.querySelector('.progress-text');
        
        progressFill.style.width = progress + '%';
        progressText.textContent = `進度: ${Math.round(progress)}%`;

        // Add celebration effect for completion
        if (progress === 100) {
            celebrateCompletion(pathSection);
        }
    }

    function celebrateCompletion(element) {
        element.classList.add('completed');
        
        // Show completion message
        const message = document.createElement('div');
        message.className = 'completion-message';
        message.innerHTML = '🎉 恭喜完成這個學習階段！';
        element.appendChild(message);

        setTimeout(function() {
            message.remove();
        }, 3000);
    }

    // ==================
    // Video Enhancements
    // ==================
    function setupVideoEnhancements() {
        const videoContainers = document.querySelectorAll('.video-container');
        
        videoContainers.forEach(function(container) {
            // Add loading indicator
            const loading = document.createElement('div');
            loading.className = 'video-loading';
            loading.innerHTML = '載入影片中...';
            container.appendChild(loading);

            // Handle iframe load
            const iframe = container.querySelector('iframe');
            if (iframe) {
                iframe.addEventListener('load', function() {
                    loading.style.display = 'none';
                });
            }

            // Add play button overlay for better UX
            const playButton = document.createElement('div');
            playButton.className = 'video-play-overlay';
            playButton.innerHTML = '▶️ 點擊播放';
            playButton.addEventListener('click', function() {
                this.style.display = 'none';
                iframe.src = iframe.src + '&autoplay=1';
            });
            container.appendChild(playButton);
        });
    }

    // ==================
    // Search Enhancements
    // ==================
    function setupSearchEnhancements() {
        const searchInput = document.querySelector('[data-md-component="search-query"]');
        if (searchInput) {
            // Add search suggestions
            setupSearchSuggestions(searchInput);
            
            // Track search analytics
            searchInput.addEventListener('input', debounce(function() {
                trackSearchQuery(this.value);
            }, 500));
        }
    }

    function setupSearchSuggestions(input) {
        const suggestions = [
            '小鼠行為分析', 'open field test', '安裝教學', 'GUI 使用',
            '參數調整', 'batch processing', 'API 文件', 'troubleshooting',
            '果蠅追蹤', '線蟲分析', 'GPU 加速', '視覺化圖表'
        ];

        input.setAttribute('placeholder', '搜尋文件... (例如: ' + 
            suggestions[Math.floor(Math.random() * suggestions.length)] + ')');
    }

    // ==================
    // Accessibility Features
    // ==================
    function setupAccessibilityFeatures() {
        // Add skip links
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'skip-link';
        skipLink.textContent = '跳至主要內容';
        document.body.insertBefore(skipLink, document.body.firstChild);

        // Improve keyboard navigation for feature cards
        const cards = document.querySelectorAll('.feature-card');
        cards.forEach(function(card) {
            card.setAttribute('tabindex', '0');
            card.setAttribute('role', 'article');
            
            card.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    this.click();
                }
            });
        });

        // Add ARIA labels to interactive elements
        enhanceAriaLabels();
    }

    function enhanceAriaLabels() {
        // Learning path checkboxes
        const checkboxes = document.querySelectorAll('.learning-path input[type="checkbox"]');
        checkboxes.forEach(function(checkbox, index) {
            if (!checkbox.getAttribute('aria-label')) {
                const label = checkbox.nextSibling ? checkbox.nextSibling.textContent.trim() : `學習項目 ${index + 1}`;
                checkbox.setAttribute('aria-label', label);
            }
        });

        // Video containers
        const videos = document.querySelectorAll('.video-container iframe');
        videos.forEach(function(video, index) {
            if (!video.getAttribute('title')) {
                video.setAttribute('title', `教學影片 ${index + 1}`);
            }
        });
    }

    // ==================
    // Interactive Elements
    // ==================
    function setupInteractiveElements() {
        // Copy code button enhancement
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('md-clipboard__message')) {
                showCopyFeedback(e.target);
            }
        });

        // Smooth scroll for anchor links
        const anchorLinks = document.querySelectorAll('a[href^="#"]');
        anchorLinks.forEach(function(link) {
            link.addEventListener('click', function(e) {
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    function showCopyFeedback(element) {
        const originalText = element.textContent;
        element.textContent = '已複製！';
        element.style.color = '#4caf50';
        
        setTimeout(function() {
            element.textContent = originalText;
            element.style.color = '';
        }, 2000);
    }

    // ==================
    // Utility Functions
    // ==================
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    function saveProgress() {
        const checkboxes = document.querySelectorAll('.learning-path input[type="checkbox"]');
        const progress = {};
        
        checkboxes.forEach(function(checkbox, index) {
            progress[`checkbox-${index}`] = checkbox.checked;
        });
        
        try {
            localStorage.setItem('castle-docs-progress', JSON.stringify(progress));
        } catch (e) {
            console.warn('Unable to save progress to localStorage');
        }
    }

    function loadProgress() {
        try {
            const saved = localStorage.getItem('castle-docs-progress');
            if (saved) {
                const progress = JSON.parse(saved);
                const checkboxes = document.querySelectorAll('.learning-path input[type="checkbox"]');
                
                checkboxes.forEach(function(checkbox, index) {
                    if (progress[`checkbox-${index}`]) {
                        checkbox.checked = true;
                        const pathSection = checkbox.closest('.learning-path');
                        if (pathSection) {
                            updatePathProgress(pathSection);
                        }
                    }
                });
            }
        } catch (e) {
            console.warn('Unable to load progress from localStorage');
        }
    }

    function trackSearchQuery(query) {
        // Track search analytics (implement with your analytics service)
        if (query.length > 2 && typeof gtag !== 'undefined') {
            gtag('event', 'search', {
                search_term: query
            });
        }
    }

    // ==================
    // Performance Monitoring
    // ==================
    window.addEventListener('load', function() {
        // Report page load performance
        if ('performance' in window) {
            const loadTime = performance.now();
            console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
            
            // Track Core Web Vitals if available
            if ('PerformanceObserver' in window) {
                trackCoreWebVitals();
            }
        }
    });

    function trackCoreWebVitals() {
        const observer = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                if (entry.entryType === 'largest-contentful-paint') {
                    console.log('LCP:', entry.startTime);
                }
                if (entry.entryType === 'first-input') {
                    console.log('FID:', entry.processingStart - entry.startTime);
                }
            });
        });
        
        try {
            observer.observe({entryTypes: ['largest-contentful-paint', 'first-input']});
        } catch (e) {
            console.warn('Performance monitoring not fully supported');
        }
    }

    // ==================
    // Error Handling
    // ==================
    window.addEventListener('error', function(e) {
        console.error('Documentation error:', e.error);
        
        // Show user-friendly error message for critical features
        if (e.target && e.target.tagName === 'IFRAME') {
            const container = e.target.closest('.video-container');
            if (container) {
                const errorMsg = document.createElement('div');
                errorMsg.className = 'video-error';
                errorMsg.innerHTML = '⚠️ 影片載入失敗，請重新整理頁面';
                container.appendChild(errorMsg);
            }
        }
    });

    // ==================
    // Export for external use
    // ==================
    window.CastleDocHelpers = {
        updatePathProgress: updatePathProgress,
        celebrateCompletion: celebrateCompletion,
        saveProgress: saveProgress,
        loadProgress: loadProgress
    };

})();