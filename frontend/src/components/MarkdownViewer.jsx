
import React, { useEffect, useRef } from 'react';
import Markdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import remarkGfm from 'remark-gfm';

const MarkdownViewer = ({ isOpen, onClose, content, highlightText, filename, variant = 'modal' }) => {
    const contentRef = useRef(null);

    // Helper to strip markdown syntax from text to match rendered HTML content
    const cleanMarkdown = (text) => {
        if (!text) return '';

        // Basic entity decoding
        const decodeEntities = (str) => {
            const textarea = document.createElement('textarea');
            textarea.innerHTML = str;
            return textarea.value;
        };

        let cleaned = text
            .replace(/^#+\s+/gm, '') // Remove headers
            .replace(/(\*\*|__)(.*?)\1/g, '$2') // Remove bold
            .replace(/(\*|_)(.*?)\1/g, '$2') // Remove italic
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove links, keep text
            .replace(/!\[([^\]]*)\]\([^)]+\)/g, '') // Remove images
            .replace(/`([^`]+)`/g, '$1') // Remove inline code
            .replace(/^\s*[-*+]\s+/gm, '') // Remove list bullets
            .replace(/^\s*\d+\.\s+/gm, '') // Remove list numbers
            .replace(/<[^>]*>/g, '') // Remove HTML tags
            .replace(/^\s*\|.*\|\s*$/gm, (match) => { // Handle tables
                if (match.match(/^\s*\|[- :|]+\|\s*$/)) return ''; // Remove separator rows
                // Preserve escaped pipes
                let row = match.replace(/\\\|/g, '___PIPE___');
                row = row.replace(/\|/g, ' '); // Replace structure pipes with spaces
                return row.replace(/___PIPE___/g, '|'); // Restore content pipes
            })
            .replace(/\n/g, ' ') // Replace newlines with spaces
            .replace(/\s+/g, ' ') // Collapse whitespace
            .trim();

        return decodeEntities(cleaned);
    };

    useEffect(() => {
        if ((isOpen || variant === 'embedded') && highlightText && contentRef.current) {
            // Wait for render
            setTimeout(() => {
                const container = contentRef.current;
                const cleanedText = cleanMarkdown(highlightText);

                if (!cleanedText) return;

                // STRATEGY: Whitespace-agnostic matching
                // 1. Strip ALL whitespace from the search string.
                // 2. Build a "DOM String" by iterating text nodes and stripping their whitespace too,
                //    while keeping a map of (DOM String Index) -> (DOM Node, Original Offset).
                // 3. Match the stripped strings.
                // 4. Use the map to look up the start/end DOM positions.

                const searchString = cleanedText.replace(/\s/g, '');
                if (!searchString) return;

                const walker = document.createTreeWalker(
                    container,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );

                let currentNode = walker.nextNode();
                let domString = "";
                const nodeMap = []; // Maps index in domString to {node, indexInNode}

                while (currentNode) {
                    const text = currentNode.nodeValue;
                    for (let i = 0; i < text.length; i++) {
                        const char = text[i];
                        if (!/\s/.test(char)) { // If not whitespace
                            domString += char;
                            nodeMap.push({ node: currentNode, index: i });
                        }
                    }
                    currentNode = walker.nextNode();
                }

                console.log("Searching for (stripped):", searchString.substring(0, 50) + "...");
                // console.log("DOM String (stripped):", domString.substring(0, 200) + "...");

                const matchIndex = domString.indexOf(searchString);

                if (matchIndex !== -1) {
                    console.log("Match found at index:", matchIndex);

                    const startIndex = matchIndex;
                    const endIndex = matchIndex + searchString.length - 1; // Inclusive

                    if (startIndex < nodeMap.length && endIndex < nodeMap.length) {
                        const startInfo = nodeMap[startIndex];
                        const endInfo = nodeMap[endIndex];

                        const range = document.createRange();
                        range.setStart(startInfo.node, startInfo.index);
                        range.setEnd(endInfo.node, endInfo.index + 1); // +1 because setEnd is exclusive

                        const selection = window.getSelection();
                        selection.removeAllRanges();
                        selection.addRange(range);

                        // Scroll into view
                        startInfo.node.parentElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                } else {
                    console.log("Could not find exact match.");
                    // Fallback: Try fuzzy match of first 20 chars
                    const fuzzyStart = searchString.substring(0, 20);
                    const fuzzyIndex = domString.indexOf(fuzzyStart);

                    if (fuzzyIndex !== -1) {
                        console.log("Found fuzzy start match");
                        const startInfo = nodeMap[fuzzyIndex];
                        startInfo.node.parentElement.scrollIntoView({ behavior: 'smooth', block: 'center' });

                        // Select the fuzzy part
                        const endInfo = nodeMap[fuzzyIndex + fuzzyStart.length - 1];
                        const range = document.createRange();
                        range.setStart(startInfo.node, startInfo.index);
                        range.setEnd(endInfo.node, endInfo.index + 1);
                        const selection = window.getSelection();
                        selection.removeAllRanges();
                        selection.addRange(range);
                    }
                }

            }, 300); // Delay to ensure Markdown is rendered
        }
    }, [isOpen, highlightText, content, variant]);

    if (!isOpen && variant === 'modal') return null;

    const contentClass = "prose prose-slate max-w-none overflow-x-auto break-words whitespace-normal prose-pre:whitespace-pre-wrap prose-pre:break-words prose-code:break-words prose-headings:text-slate-800 prose-p:text-slate-600 prose-a:text-[var(--color-primary)] prose-strong:text-slate-800 prose-table:border-collapse prose-table:border prose-table:border-slate-200 prose-table:shadow-sm prose-th:border prose-th:border-slate-200 prose-th:bg-slate-50 prose-th:p-3 prose-th:text-slate-700 prose-td:border prose-td:border-slate-200 prose-td:p-3 prose-tr:even:bg-slate-50/50";

    if (variant === 'embedded') {
        return (
            <div className="h-full flex flex-col bg-white/70 backdrop-blur-md rounded-xl border border-white/50 shadow-sm overflow-hidden">
                <div className="flex-1 overflow-y-auto p-6 bg-transparent">
                    <div className={`${contentClass} w-full`} ref={contentRef}>
                        <Markdown
                            rehypePlugins={[rehypeRaw, rehypeSanitize]}
                            remarkPlugins={[remarkGfm]}
                        >
                            {content}
                        </Markdown>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 z-[2000] flex items-center justify-center p-2 sm:p-4 bg-black/10 backdrop-blur-sm animate-in fade-in duration-200">
            <div
                className="glass-panel rounded-2xl shadow-2xl w-full max-w-[1000px] h-[92vh] flex flex-col overflow-hidden animate-in zoom-in-95 duration-200"
                style={{ backgroundColor: 'rgba(255, 255, 255, 0.4)' }}
            >
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-white/60 bg-white/70 backdrop-blur-md sticky top-0 z-10 shrink-0 min-w-0">
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                        <div className="p-2 bg-[var(--color-primary)] text-white rounded-lg shadow-sm shrink-0">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                        </div>
                        <div className="min-w-0 flex-1">
                            <h3 className="font-bold text-slate-800 text-[13px] leading-tight truncate">{filename}</h3>
                            <p className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">Source Document</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="btn btn-sm btn-secondary flex items-center gap-2 px-3 py-1.5 rounded-lg border-white/60 bg-white/40 hover:bg-white/60 transition-all font-bold text-[11px] uppercase tracking-wider"
                        title="Close Viewer"
                    >
                        <span>Close</span>
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>

                {/* Content - White Page Area */}
                <div className="flex-1 overflow-y-auto p-6 md:p-10 bg-white min-w-0">
                    <div className={`${contentClass} w-full`} ref={contentRef}>
                        <Markdown
                            rehypePlugins={[rehypeRaw, rehypeSanitize]}
                            remarkPlugins={[remarkGfm]}
                            components={{
                                pre: ({ node: _node, ...props }) => <pre {...props} className="whitespace-pre-wrap break-words overflow-x-hidden bg-slate-50 border border-slate-200 rounded-lg p-3" />,
                                code: ({ node: _node, ...props }) => <code {...props} className="whitespace-pre-wrap break-words font-medium text-[var(--color-primary)] bg-transparent p-0" />
                            }}
                        >
                            {content}
                        </Markdown>
                    </div>
                </div>

                {/* Footer */}
                <div className="px-4 py-2.5 bg-white/60 backdrop-blur-md border-t border-white/40 text-[10px] font-bold text-slate-400 flex justify-between items-center shrink-0 min-w-0 uppercase tracking-widest">
                    <span className="truncate mr-2">Markdown View</span>
                    <span className="whitespace-nowrap">{content ? content.length.toLocaleString() : 0} chars</span>
                </div>
            </div>
        </div>
    );
};

export default MarkdownViewer;
