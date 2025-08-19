// ThebeLab configuration for ProbLang v2 site
$(document).ready(function() {
    let isKernelReady = false;
    let kernelRequestInProgress = false;
    let executingCells = new Set(); // Track cells currently executing
    
    // Configure ThebeLab options (but don't bootstrap yet)
    const thebeConfig = {
        requestKernel: true,
        binderOptions: {
            repo: "SocialInteractionLab/ProbLang"
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
    
    // Function to enhance run button appearance
    function enhanceRunButtons() {
        $('.thebe-run-button').each(function() {
            const $button = $(this);
            if (!$button.find('.button-text').length) {
                const originalText = $button.text();
                $button.html('<span class="button-text">' + originalText + '</span>');
            }
        });
    }
    
    // Function to set button to running state
    function setButtonRunning($button) {
        $button.addClass('running');
        $button.find('.button-text').text('Running...');
        $button.prop('disabled', true);
    }
    
    // Function to set button to normal state
    function setButtonNormal($button) {
        $button.removeClass('running success');
        $button.find('.button-text').text('Run');
        $button.prop('disabled', false);
    }
    
    // Function to set button to success state briefly
    function setButtonSuccess($button) {
        $button.removeClass('running');
        $button.addClass('success');
        $button.find('.button-text').text('Done!');
        
        // Return to normal after animation
        setTimeout(() => {
            setButtonNormal($button);
        }, 1000);
    }
    
    // Function to disable all thebe run buttons
    function disableThebeButtons() {
        // Disable Thebe's built-in run buttons
        $('.thebe-run-button').prop('disabled', true);
        
        // Add a visual indicator and tooltip
        $('.thebe-run-button').attr('title', 'Please activate the kernel first');
    }
    
    // Function to enable all thebe run buttons
    function enableThebeButtons() {
        $('.thebe-run-button').prop('disabled', false);
        
        // Update tooltips
        $('.thebe-run-button').attr('title', 'Run this cell');
        
        // Enhance button appearance
        setTimeout(enhanceRunButtons, 100);
    }
    
    // Function to show user-friendly message when buttons are clicked before kernel is ready
    function showKernelNotReadyMessage() {
        // Create a temporary notification
        const $message = $('<div class="kernel-not-ready-message">Please wait for the kernel to start automatically.</div>');
        $message.css({
            position: 'fixed',
            top: '60px',
            right: '10px',
            background: '#ffc107',
            color: '#856404',
            padding: '10px 15px',
            borderRadius: '5px',
            zIndex: 1001,
            fontSize: '14px',
            maxWidth: '300px',
            boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
        });
        
        $('body').append($message);
        
        // Remove message after 4 seconds
        setTimeout(() => {
            $message.fadeOut(() => $message.remove());
        }, 4000);
    }
    
    // Function to simulate execution completion (since we can't easily hook into Thebe's execution events)
    function simulateExecutionCompletion($button, cellId) {
        let cellElement = $button.closest('[data-thebe-executable]');
        let hasDetectedCompletion = false;
        
        // Debug: Check if we found the cell element
        if (cellElement.length === 0) {
            console.warn('Could not find cell element with [data-thebe-executable]');
            // Fallback: try to find the parent container
            cellElement = $button.closest('.highlight, .thebe-cell, pre');
            if (cellElement.length > 0) {
                console.log('Using alternative cell element:', cellElement[0]);
            } else {
                console.error('Could not find any valid cell element - using timeout only');
                setTimeout(() => {
                    executingCells.delete(cellId);
                    setButtonNormal($button);
                }, 10000); // 10 second fallback
                return;
            }
        }
        
        // More comprehensive output detection
        const checkForOutput = () => {
            // Look for various types of Thebe output containers
            const outputs = cellElement.find('.thebe-output, .jp-OutputArea, .output_area, [class*="output"]');
            const hasOutput = outputs.length > 0;
            
            // Check for error indicators
            const hasError = cellElement.find('.error, .jp-RenderedText[data-mime-type*="error"], .output_error').length > 0;
            
            // Check if there's any text content in output areas (successful execution)
            const hasTextOutput = outputs.filter(function() {
                return $(this).text().trim().length > 0;
            }).length > 0;
            
            return hasOutput && (hasTextOutput || hasError);
        };
        
        // Use MutationObserver for more reliable detection
        const observer = new MutationObserver(function(mutations) {
            if (hasDetectedCompletion) return;
            
            mutations.forEach(function(mutation) {
                // Check if any new nodes were added that might be output
                if (mutation.addedNodes.length > 0) {
                    Array.from(mutation.addedNodes).forEach(function(node) {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            const $node = $(node);
                            // Check if this is an output-related element
                            if ($node.hasClass('thebe-output') || 
                                $node.hasClass('jp-OutputArea') || 
                                $node.hasClass('output_area') ||
                                $node.find('.thebe-output, .jp-OutputArea, .output_area').length > 0 ||
                                ($node.text() && $node.text().trim().length > 0)) {
                                
                                console.log('Detected output completion via MutationObserver');
                                completeExecution(false);
                            }
                        }
                    });
                }
            });
        });
        
        // Observe the cell for changes (with safety check)
        if (cellElement.length > 0 && cellElement[0] instanceof Node) {
            observer.observe(cellElement[0], {
                childList: true,
                subtree: true,
                characterData: true
            });
        } else {
            console.warn('Unable to observe cell element - falling back to polling only');
        }
        
        // Function to complete execution
        const completeExecution = (isTimeout = false) => {
            if (hasDetectedCompletion) return;
            hasDetectedCompletion = true;
            
            observer.disconnect();
            executingCells.delete(cellId);
            
            if (isTimeout) {
                console.log('Execution completed via timeout');
                setButtonNormal($button);
            } else {
                // Check for errors one more time
                const hasError = cellElement.find('.error, .jp-RenderedText[data-mime-type*="error"], .output_error').length > 0;
                if (hasError) {
                    $button.removeClass('running');
                    $button.find('.button-text').text('Error');
                    setTimeout(() => setButtonNormal($button), 1500);
                } else {
                    setButtonSuccess($button);
                }
            }
        };
        
        // Polling fallback for cases where MutationObserver might miss changes
        let pollCount = 0;
        const maxPolls = 60; // 30 seconds of polling
        
        const pollForCompletion = () => {
            if (hasDetectedCompletion) return;
            
            pollCount++;
            
            if (checkForOutput()) {
                console.log('Detected output completion via polling');
                completeExecution(false);
                return;
            }
            
            if (pollCount >= maxPolls) {
                console.log('Execution timed out after polling');
                completeExecution(true);
                return;
            }
            
            setTimeout(pollForCompletion, 500);
        };
        
        // Start polling after a brief delay
        setTimeout(pollForCompletion, 1000);
        
        // Ultimate fallback timeout
        setTimeout(() => {
            if (!hasDetectedCompletion) {
                console.log('Execution completed via ultimate timeout');
                completeExecution(true);
            }
        }, 35000); // 35 second ultimate timeout
    }
    
    // Override Thebe's click handlers to prevent execution before kernel is ready
    function interceptThebeButtons() {
        // Use event delegation to catch button clicks
        $(document).on('click', '.thebe-run-button', function(e) {
            const $button = $(this);
            
            if (!isKernelReady) {
                e.preventDefault();
                e.stopPropagation();
                showKernelNotReadyMessage();
                return false;
            }
            
            // Prevent multiple executions of the same cell
            const cellElement = $button.closest('[data-thebe-executable]');
            const cellId = cellElement.attr('id') || cellElement.index();
            
            if (executingCells.has(cellId)) {
                e.preventDefault();
                e.stopPropagation();
                
                // Show brief "already executing" message
                const $execMessage = $('<div class="execution-in-progress-message">Cell is already executing...</div>');
                $execMessage.css({
                    position: 'fixed',
                    top: '60px',
                    right: '10px',
                    background: '#17a2b8',
                    color: 'white',
                    padding: '8px 12px',
                    borderRadius: '3px',
                    zIndex: 1001,
                    fontSize: '12px'
                });
                
                $('body').append($execMessage);
                setTimeout(() => {
                    $execMessage.fadeOut(() => $execMessage.remove());
                }, 2000);
                
                return false;
            }
            
            // Mark cell as executing and set visual state
            executingCells.add(cellId);
            setButtonRunning($button);
            
            // Set up completion detection
            simulateExecutionCompletion($button, cellId);
            
            // Fallback: clear executing state after maximum time
            setTimeout(() => {
                if (executingCells.has(cellId)) {
                    executingCells.delete(cellId);
                    setButtonNormal($button);
                }
            }, 30000); // 30 second timeout
        });
    }
    
    // Initialize status button to show kernel state
    const $statusButton = $('<button id="thebe-activate-button" class="thebe-activate-button">🔄 Starting Kernel...</button>');
    $statusButton.prop('disabled', true);
    
    // Function to automatically start the kernel
    function autoStartKernel() {
        if (kernelRequestInProgress) {
            console.log('Kernel request already in progress, skipping');
            return; // Prevent multiple kernel requests
        }
        
        console.log('Starting kernel bootstrap process...');
        
        // Check if thebelab is available
        if (typeof thebelab === 'undefined') {
            console.error('ThebeLab library not loaded yet, retrying in 1 second...');
            setTimeout(() => autoStartKernel(), 1000);
            return;
        }
        
        kernelRequestInProgress = true;
        $statusButton.text('🚀 Starting Kernel...').prop('disabled', true);
        
        console.log('Calling thebelab.bootstrap with config:', thebeConfig);
        thebelab.bootstrap(thebeConfig).then(function() {
            console.log('ThebeLab kernel started successfully');
            isKernelReady = true;
            kernelRequestInProgress = false;
            $statusButton.text('✅ Kernel Ready').css('background-color', '#28a745');
            
            // Enable the thebe buttons
            enableThebeButtons();
            
            // Store kernel reference - try different API access patterns
            if (thebelab && thebelab.manager && thebelab.manager.kernel) {
                window.thebeKernel = thebelab.manager.kernel;
                console.log('Kernel stored via manager');
            } else if (thebelab && thebelab.kernel) {
                window.thebeKernel = thebelab.kernel;
                console.log('Kernel stored via direct access');
            } else {
                console.log('Kernel reference not found, but ThebeLab is ready');
                window.thebeKernel = null;
            }
            
        }).catch(function(error) {
            console.error('Failed to start ThebeLab kernel:', error);
            kernelRequestInProgress = false;
            
            // Check if this is a CORS error (common in localhost development)
            if (error.message && error.message.includes('CORS')) {
                $statusButton.text('⚠️ CORS Issue (localhost)').prop('disabled', false)
                          .css('background-color', '#ffc107')
                          .attr('title', 'CORS error - try on production site or click to retry');
            } else {
                $statusButton.text('❌ Kernel Failed - Click to Retry').prop('disabled', false)
                          .css('background-color', '#dc3545');
            }
        });
    }
    
    // Allow manual retry if auto-start fails
    $statusButton.click(function() {
        if (!isKernelReady && !kernelRequestInProgress) {
            autoStartKernel();
        }
    });
    
    // Add the status button to pages with executable code
    const executableElements = $("[data-thebe-executable='true']");
    console.log('Found executable elements:', executableElements.length);
    
    if (executableElements.length > 0) {
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
}); 
