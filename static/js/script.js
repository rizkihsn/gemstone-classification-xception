document.addEventListener('DOMContentLoaded', () => {
    // 1. Scroll effect for Navbar
    const navbar = document.querySelector('.glass-nav');
    if (navbar) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }

    // 2. Intersection Observer for scroll animations (fade-up, fade-left)
    const animatedElements = document.querySelectorAll('.fade-up, .fade-left');
    
    if ('IntersectionObserver' in window) {
        const observerOptions = {
            root: null,
            rootMargin: '0px 0px -50px 0px',
            threshold: 0
        };

        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    obs.unobserve(entry.target);
                }
            });
        }, observerOptions);

        animatedElements.forEach(el => observer.observe(el));
        
        // Force check immediately in case elements are already in viewport
        setTimeout(() => {
            animatedElements.forEach(el => {
                const rect = el.getBoundingClientRect();
                if (rect.top < window.innerHeight && rect.bottom > 0) {
                    el.classList.add('visible');
                }
            });
        }, 50);
    } else {
        // Fallback
        animatedElements.forEach(el => el.classList.add('visible'));
    }

    // 3. Drag and Drop Logic for predict.html
    const uploadZone = document.getElementById('uploadZone');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadIcon = document.getElementById('uploadIcon');
    const uploadText = document.getElementById('uploadText');
    const submitBtn = document.getElementById('submitBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const predictionForm = document.getElementById('predictionForm');
    const resetBtn = document.getElementById('resetBtn');

    if (uploadZone && imageInput) {
        // Click to open file dialog
        uploadZone.addEventListener('click', () => {
            imageInput.click();
        });

        // Drag events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => {
                uploadZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => {
                uploadZone.classList.remove('dragover');
            }, false);
        });

        uploadZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        });

        imageInput.addEventListener('change', function() {
            handleFiles(this.files);
        });

        function handleFiles(files) {
            if (files.length === 0) return;
            
            const file = files[0];
            const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
            
            if (!validTypes.includes(file.type)) {
                alert('Please upload a valid image file (JPG, JPEG, PNG).');
                resetUpload();
                return;
            }

            // Preview image
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('d-none');
                uploadIcon.classList.add('d-none');
                uploadText.classList.add('d-none');
                submitBtn.disabled = false;
                if(resetBtn) resetBtn.classList.remove('d-none');
            }
            reader.readAsDataURL(file);

            // Update input files object if drag-dropped
            if (imageInput.files !== files) {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                imageInput.files = dataTransfer.files;
            }
        }

        function resetUpload() {
            imageInput.value = '';
            imagePreview.src = '#';
            imagePreview.classList.add('d-none');
            uploadIcon.classList.remove('d-none');
            uploadText.classList.remove('d-none');
            submitBtn.disabled = true;
            if(resetBtn) resetBtn.classList.add('d-none');
        }

        if(resetBtn) {
            resetBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // prevent clicking upload zone
                resetUpload();
            });
        }

        // Form submission loading state
        if (predictionForm) {
            predictionForm.addEventListener('submit', (e) => {
                if (imageInput.files.length > 0) {
                    if (loadingOverlay) loadingOverlay.classList.remove('d-none');
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Processing...';
                }
            });
        }
    }

    // 4. Animate Confidence Bar in result.html
    const confidenceFill = document.querySelector('.confidence-fill');
    if (confidenceFill) {
        const targetWidth = confidenceFill.getAttribute('data-width');
        // Small delay for smooth entry animation
        setTimeout(() => {
            confidenceFill.style.width = targetWidth;
            
            // Adjust color based on confidence
            const confidenceValue = parseFloat(targetWidth);
            if (confidenceValue >= 80) {
                confidenceFill.style.background = 'linear-gradient(90deg, #10B981, #34D399)'; // Green
            } else if (confidenceValue >= 50) {
                confidenceFill.style.background = 'linear-gradient(90deg, #F59E0B, #FBBF24)'; // Yellow
            } else {
                confidenceFill.style.background = 'linear-gradient(90deg, #EF4444, #F87171)'; // Red
            }
        }, 300);
    }
});
