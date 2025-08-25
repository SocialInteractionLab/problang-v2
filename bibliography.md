---
layout: default
title: Bibliography
permalink: /bibliography/
---

# Bibliography

This page contains all the references cited throughout the book.

<div id="bibliography-content">
  <p>Loading bibliography...</p>
</div>

<script>
// Function to format a single citation for display
function formatCitationForDisplay(citation) {
    var html = '<div class="citation-entry">';
    
    // Authors
    if (citation["AUTHOR"]) {
        html += '<div class="citation-authors">' + textohtml(citation["AUTHOR"]) + '</div>';
    }
    
    // Title
    if (citation["TITLE"]) {
        if (citation["URL"]) {
            html += '<div class="citation-title"><a href="' + citation["URL"] + '" target="_blank" rel="noopener">' + textohtml(citation["TITLE"]) + '</a></div>';
        } else {
            html += '<div class="citation-title">' + textohtml(citation["TITLE"]) + '</div>';
        }
    }
    
    // Publication details
    var pubDetails = [];
    if (citation["JOURNAL"]) {
        pubDetails.push('<em>' + textohtml(citation["JOURNAL"]) + '</em>');
    }
    if (citation["BOOKTITLE"]) {
        pubDetails.push('<em>' + textohtml(citation["BOOKTITLE"]) + '</em>');
    }
    if (citation["YEAR"]) {
        pubDetails.push(citation["YEAR"]);
    }
    if (citation["VOLUME"]) {
        pubDetails.push('Vol. ' + citation["VOLUME"]);
    }
    if (citation["PAGES"]) {
        pubDetails.push('pp. ' + citation["PAGES"]);
    }
    if (citation["PUBLISHER"]) {
        pubDetails.push(textohtml(citation["PUBLISHER"]));
    }
    
    if (pubDetails.length > 0) {
        html += '<div class="citation-details">' + pubDetails.join(', ') + '</div>';
    }
    
    // DOI or URL
    if (citation["DOI"]) {
        html += '<div class="citation-doi"><strong>DOI:</strong> <a href="https://doi.org/' + citation["DOI"] + '" target="_blank" rel="noopener">' + citation["DOI"] + '</a></div>';
    } else if (citation["URL"] && !citation["TITLE"]) {
        html += '<div class="citation-url"><strong>URL:</strong> <a href="' + citation["URL"] + '" target="_blank" rel="noopener">' + citation["URL"] + '</a></div>';
    }
    
    html += '</div>';
    return html;
}

// Load and display bibliography
$.get("/problang-v2/bibliography.bib", function (bibtext) {
    var bibs = doParse(bibtext);
    var citationEntries = [];
    
    $.each(bibs, function (citation_id, citation) {
        // Skip if citation_id starts with @ or if citation is invalid
        if (citation_id.startsWith('@') || !citation || !citation["AUTHOR"]) {
            return;
        }
        
        var formattedCitation = formatCitationForDisplay(citation);
        citationEntries.push({
            id: citation_id,
            html: formattedCitation,
            author: citation["AUTHOR"] || "",
            year: citation["YEAR"] || "0000"
        });
    });
    
    // Sort by author last name, then year
    citationEntries.sort(function(a, b) {
        var authorA = a.author.split(',')[0].toLowerCase();
        var authorB = b.author.split(',')[0].toLowerCase();
        if (authorA !== authorB) {
            return authorA.localeCompare(authorB);
        }
        return b.year.localeCompare(a.year); // Newer first for same author
    });
    
    // Generate HTML
    var html = '<div class="bibliography-entries">';
    citationEntries.forEach(function(entry) {
        html += entry.html;
    });
    html += '</div>';
    
    $('#bibliography-content').html(html);
    
}).fail(function(jqXHR, textStatus, errorThrown) {
    console.error('Failed to load bibliography:', textStatus, errorThrown);
    $('#bibliography-content').html('<p class="error">Failed to load bibliography. Please try refreshing the page.</p>');
});
</script>

<style>
.bibliography-entries {
    max-width: 800px;
    margin: 0 auto;
}

.citation-entry {
    margin-bottom: 1.5em;
    padding: 1em;
    border: 1px solid #e5e5e5;
    border-radius: 4px;
    background-color: #fafafa;
}

.citation-authors {
    font-weight: bold;
    margin-bottom: 0.5em;
    color: #333;
}

.citation-title {
    font-size: 1.1em;
    margin-bottom: 0.5em;
    line-height: 1.4;
}

.citation-title a {
    color: #2563eb;
    text-decoration: none;
}

.citation-title a:hover {
    text-decoration: underline;
}

.citation-details {
    color: #666;
    margin-bottom: 0.5em;
    font-style: italic;
}

.citation-doi, .citation-url {
    font-size: 0.9em;
    color: #666;
}

.citation-doi a, .citation-url a {
    color: #2563eb;
    text-decoration: none;
}

.citation-doi a:hover, .citation-url a:hover {
    text-decoration: underline;
}

.error {
    color: #dc2626;
    font-weight: bold;
    text-align: center;
    padding: 2em;
}

@media (max-width: 768px) {
    .citation-entry {
        margin: 0 0 1em 0;
        padding: 0.75em;
    }
    
    .citation-title {
        font-size: 1em;
    }
}
</style> 