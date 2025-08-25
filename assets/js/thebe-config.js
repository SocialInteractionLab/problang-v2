// ThebeLab configuration for ProbLang v2 site
$(document).ready(function() {
    let isKernelReady = false;
    let kernelRequestInProgress = false;
    let executingCells = new Set(); // Track cells currently executing
    
    // Add comprehensive debugging
    console.log('=== THEBE DEBUGGING START ===');
    console.log('Document ready, checking for Thebe libraries...');
    
    // Check what libraries are available
    console.log('typeof thebe:', typeof thebe);
    console.log('typeof thebelab:', typeof thebelab);
    console.log('window.thebe:', window.thebe);
    console.log('window.thebelab:', window.thebelab);
    
    // Configure ThebeLab options (but don't bootstrap yet)
    const thebeConfig = {
        requestKernel: true,
        binderOptions: {
            repo: "SocialInteractionLab/problang-v2",
            ref: "main",
            binderUrl: "https://mybinder.org"
        },
        kernelOptions: {
            name: "python3"
        },
        selector: "[data-thebe-executable='true']",
        mountRunButton: true,
        mountRunAllButton: false,
        mountRestartButton: false,
        mountRestartAllButton: false
    };
    
    console.log('Thebe config:', thebeConfig);
    
    // ... existing code ...
    
    // Function to automatically start the kernel
    function autoStartKernel() {
        if (kernelRequestInProgress) {
            console.log('Kernel request already in progress, skipping');
            return; // Prevent multiple kernel requests
        }
        
        console.log('=== STARTING KERNEL BOOTSTRAP PROCESS ===');
        console.log('Current time:', new Date().toISOString());
        
        // Check if thebelab is available
        if (typeof thebe === 'undefined' && typeof thebelab === 'undefined') {
            console.error('Neither thebe nor thebelab is available!');
            console.log('Available global objects:', Object.keys(window).filter(k => k.includes('thebe')));
            console.log('Scripts loaded:', document.querySelectorAll('script[src*="thebe"]));
            $statusButton.text('❌ Thebe not loaded').prop('disabled', false)
                      .css('background-color', '#dc3545');
            return;
        }
        
        kernelRequestInProgress = true;
        $statusButton.text('�� Starting Kernel...').prop('disabled', true);
        
        console.log('Calling bootstrap with config:', thebeConfig);
        
        // Try the newer Thebe API first
        if (typeof thebe !== 'undefined') {
            console.log('Using newer Thebe API...');
            thebe.bootstrap(thebeConfig).then(function(result) {
                console.log('=== THEBE BOOTSTRAP SUCCESS ===');
                console.log('Bootstrap result:', result);
                console.log('Thebe object after bootstrap:', thebe);
                
                isKernelReady = true;
                kernelRequestInProgress = false;
                $statusButton.text('✅ Kernel Ready').css('background-color', '#28a745');
                
                // Enable the thebe buttons
                enableThebeButtons();
                
                // Store kernel reference
                if (thebe && thebe.manager && thebe.manager.kernel) {
                    window.thebeKernel = thebe.manager.kernel;
                    console.log('Kernel stored via manager:', window.thebeKernel);
                } else if (thebe && thebe.kernel) {
                    window.thebeKernel = thebe.kernel;
                    console.log('Kernel stored via direct access:', window.thebeKernel);
                } else {
                    console.log('Kernel reference not found, but Thebe is ready');
                    console.log('Available thebe properties:', Object.keys(thebe));
                    window.thebeKernel = null;
                }
                
            }).catch(function(error) {
                console.error('=== THEBE BOOTSTRAP FAILED ===');
                console.error('Error details:', error);
                console.error('Error message:', error.message);
                console.error('Error stack:', error.stack);
                console.error('Error name:', error.name);
                console.error('Full error object:', JSON.stringify(error, null, 2));
                
                // Fall back to thebelab if thebe fails
                console.log('Falling back to ThebeLab API...');
                fallbackToThebeLab();
            });
        } else if (typeof thebelab !== 'undefined') {
            // Fall back to the older ThebeLab API
            console.log('Using ThebeLab API...');
            fallbackToThebeLab();
        }
        
        function fallbackToThebeLab() {
            console.log('=== FALLBACK TO THEBELAB ===');
            console.log('ThebeLab object:', thebelab);
            console.log('ThebeLab methods:', Object.getOwnPropertyNames(thebelab));
            
            thebelab.bootstrap(thebeConfig).then(function(result) {
                console.log('=== THEBELAB BOOTSTRAP SUCCESS ===');
                console.log('Bootstrap result:', result);
                console.log('ThebeLab object after bootstrap:', thebelab);
                
                isKernelReady = true;
                kernelRequestInProgress = false;
                $statusButton.text('✅ Kernel Ready').css('background-color', '#28a745');
                
                // Enable the thebe buttons
                enableThebeButtons();
                
                // Store kernel reference - try different API access patterns
                if (thebelab && thebelab.manager && thebelab.manager.kernel) {
                    window.thebeKernel = thebelab.manager.kernel;
                    console.log('Kernel stored via manager:', window.thebeKernel);
                } else if (thebelab && thebelab.kernel) {
                    window.thebeKernel = thebelab.kernel;
                    console.log('Kernel stored via direct access:', window.thebeKernel);
                } else {
                    console.log('Kernel reference not found, but ThebeLab is ready');
                    console.log('Available thebelab properties:', Object.keys(thebelab));
                    window.thebeKernel = null;
                }
                
            }).catch(function(error) {
                console.error('=== THEBELAB BOOTSTRAP FAILED ===');
                console.error('Error details:', error);
                console.error('Error message:', error.message);
                console.error('Error stack:', error.stack);
                console.error('Error name:', error.name);
                console.error('Full error object:', JSON.stringify(error, null, 2));
                
                kernelRequestInProgress = false;
                
                // Check if this is a CORS error (common in localhost development)
                if (error.message && error.message.includes('CORS')) {
                    console.log('CORS error detected');
                    $statusButton.text('⚠️ CORS Issue (localhost)').prop('disabled', false)
                              .css('background-color', '#ffc107')
                              .attr('title', 'CORS error - try on production site or click to retry');
                } else {
                    console.log('Non-CORS error detected');
                    $statusButton.text('❌ Kernel Failed - Click to Retry').prop('disabled', false)
                              .css('background-color', '#dc3545');
                }
            });
        }
    }
    
    // ... existing code ...
    
    // Add the status button to pages with executable code
    const executableElements = $("[data-thebe-executable='true']");
    console.log('Found executable elements:', executableElements.length);
    
    if (executableElements.length > 0) {
        console.log('=== SETTING UP THEBE ===');
        $('body').prepend($statusButton);
        
        // Disable thebe buttons initially and set up interception
        setTimeout(() => {
            disableThebeButtons();
            interceptThebeButtons();
            enhanceRunButtons();
        }, 1000); // Give thebe time to render buttons
        
        // Check if we should auto-start or wait for manual activation
        const isLocalhost = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
        console.log('Current hostname:', window.location.hostname, 'isLocalhost:', isLocalhost);
        
        if (isLocalhost) {
            // On localhost, provide manual activation due to potential CORS issues
            console.log('Localhost detected - enabling manual activation');
            $statusButton.text('🔌 Click to Start Kernel')
                       .prop('disabled', false)
                       .attr('title', 'Click to start kernel (auto-start disabled on localhost due to CORS)');
        } else {
            // On production, auto-start the kernel
            console.log('Production environment detected - enabling auto-start');
            setTimeout(() => {
                console.log('Auto-starting kernel...');
                autoStartKernel();
            }, 500);
        }
    } else {
        console.log('No executable elements found on this page - Thebe not initialized');
    }
    
    console.log('=== THEBE DEBUGGING END ===');
});