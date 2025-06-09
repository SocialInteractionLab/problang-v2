// ThebeLab configuration for neuroPrag site
$(document).ready(function() {
    let isKernelReady = false;
    
    // Configure ThebeLab options (but don't bootstrap yet)
    const thebeConfig = {
        // Use Binder as the kernel provider
        binderOptions: {
            repo: "hawkrobe/probLang-memo",  // Correct GitHub repository
            ref: "main",
            binderUrl: "https://mybinder.org",
            // Specify which environment file to use (environment.yml takes precedence)
        },
        // Kernel options
        kernelOptions: {
            name: "python3",
            path: "."
        },
        // Selector for code cells
        selector: "[data-thebe-executable='true']",
        // Options for the code cells
        requestKernel: true,
        predefinedOutput: true,
        mountActivateWidget: true,
        mountStatusWidget: true
    };
    
    // Add run buttons to executable code blocks
    $("[data-executable='true']").each(function() {
        const $this = $(this);
        const $runButton = $('<button class="thebe-run-button" disabled>▶ Run (Activate kernel first)</button>');
        $runButton.click(function() {
            if (isKernelReady) {
                // Enable ThebeLab on this cell and execute
                const codeElement = $this.find('code')[0];
                if (codeElement) {
                    // Set ThebeLab attributes to make it executable
                    $this.attr('data-thebe-executable', 'true');
                    
                    // Try to trigger execution using ThebeLab's method
                    try {
                        thebelab.run(codeElement);
                        console.log('Code execution triggered');
                    } catch (error) {
                        console.warn('Direct execution failed, trying alternative:', error);
                        
                        // Fallback: create a ThebeLab cell manually
                        const code = codeElement.textContent;
                        if (window.thebeKernel && window.thebeKernel.execute) {
                            window.thebeKernel.execute(code).then(function(result) {
                                console.log('Execution result:', result);
                            });
                        } else {
                            console.error('No execution method available');
                        }
                    }
                }
            }
        });
        $this.before($runButton);
    });
    
    // Initialize button to start the kernel
    const $initButton = $('<button id="thebe-activate-button" class="thebe-activate-button">Activate Interactive Code</button>');
    $initButton.click(function() {
        $(this).text('🚀 Starting Kernel...').prop('disabled', true);
        
        thebelab.bootstrap(thebeConfig).then(function() {
            console.log('ThebeLab kernel started successfully');
            isKernelReady = true;
            $initButton.text('✅ Kernel Ready').css('background-color', '#28a745');
            
            // Enable run buttons
            $('.thebe-run-button').prop('disabled', false).text('▶ Run');
            
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
            $initButton.text('❌ Kernel Failed - Try Again').prop('disabled', false)
                      .css('background-color', '#dc3545');
        });
    });
    
    // Add the activate button to pages with executable code
    if ($("[data-executable='true']").length > 0) {
        $('body').prepend($initButton);
    }
}); 