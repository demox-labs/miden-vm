// use talc::*;

// static mut ARENA: [u8; 536870911] = [0; 536870911];

// #[global_allocator]
// static ALLOCATOR: Talck<spin::Mutex<()>, ClaimOnOom> = Talc::new(unsafe {
//     // if we're in a hosted environment, the Rust runtime may allocate before
//     // main() is called, so we need to initialize the arena automatically
//     ClaimOnOom::new(Span::from_const_array(core::ptr::addr_of!(ARENA)))
// }).lock();

use super::{
    ast::{instrument, Instruction, ModuleAst, Node, ProcedureAst, ProgramAst},
    crypto::hash::RpoDigest,
    AssemblyError, CallSet, CodeBlock, CodeBlockTable, Felt, Kernel, Library, LibraryError,
    LibraryPath, Module, NamedProcedure, Operation, Procedure, ProcedureId, ProcedureName, Program,
    ONE, ZERO,
};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::{borrow::Borrow, cell::RefCell};
use vm_core::{utils::group_vector_elements, Decorator, DecoratorList};

mod instruction;

mod module_provider;
use module_provider::ModuleProvider;

mod span_builder;
use span_builder::SpanBuilder;

mod context;
pub use context::AssemblyContext;

mod procedure_cache;
use procedure_cache::ProcedureCache;

#[cfg(test)]
mod tests;

// ASSEMBLER
// ================================================================================================
/// Miden Assembler which can be used to convert Miden assembly source code into program MAST.
///
/// The assembler can be instantiated in several ways using a "builder" pattern. Specifically:
/// - If `with_kernel()` or `with_kernel_module()` methods are not used, the assembler will be
///   instantiated with a default empty kernel. Programs compiled using such assembler
///   cannot make calls to kernel procedures via `syscall` instruction.
#[derive(Default)]
pub struct Assembler {
    kernel: Kernel,
    module_provider: ModuleProvider,
    proc_cache: RefCell<ProcedureCache>,
    in_debug_mode: bool,
}

impl Assembler {
    // CONSTRUCTORS
    // --------------------------------------------------------------------------------------------

    /// Puts the assembler into the debug mode.
    pub fn with_debug_mode(mut self, in_debug_mode: bool) -> Self {
        self.in_debug_mode = in_debug_mode;
        self
    }

    /// Adds the library to provide modules for the compilation.
    pub fn with_library<L>(mut self, library: &L) -> Result<Self, AssemblyError>
    where
        L: Library,
    {
        self.module_provider.add_library(library)?;
        Ok(self)
    }

    /// Adds a library bundle to provide modules for the compilation.
    pub fn with_libraries<I, L>(self, mut libraries: I) -> Result<Self, AssemblyError>
    where
        L: Library,
        I: Iterator<Item = L>,
    {
        libraries.try_fold(self, |slf, library| slf.with_library(&library))
    }

    /// Sets the kernel for the assembler to the kernel defined by the provided source.
    ///
    /// # Errors
    /// Returns an error if compiling kernel source results in an error.
    ///
    /// # Panics
    /// Panics if the assembler has already been used to compile programs.
    pub fn with_kernel(self, kernel_source: &str) -> Result<Self, AssemblyError> {
        web_sys::console::log_1(&"with kernel 1".into());
        let kernel_ast = ModuleAst::parse(kernel_source)?;
        web_sys::console::log_1(&"with kernel 2".into());
        self.with_kernel_module(kernel_ast)
    }

    /// Sets the kernel for the assembler to the kernel defined by the provided module.
    ///
    /// # Errors
    /// Returns an error if compiling kernel source results in an error.
    pub fn with_kernel_module(mut self, module: ModuleAst) -> Result<Self, AssemblyError> {
        // compile the kernel; this adds all exported kernel procedures to the procedure cache
        // console_error_panic_hook::set_once();
        web_sys::console::log_1(&"with kernel module".into());
        let mut context = AssemblyContext::for_module(true);
        web_sys::console::log_1(&"with kernel module 1".into());
        let kernel = Module::kernel(module);
        web_sys::console::log_1(&"with kernel module 2".into());
        self.compile_module(&kernel.ast, Some(&kernel.path), &mut context)?;
        web_sys::console::log_1(&"with kernel module 30".into());

        // convert the context into Kernel; this builds the kernel from hashes of procedures
        // exported form the kernel module
        self.kernel = context.into_kernel();
        web_sys::console::log_1(&"with kernel module 4".into());

        Ok(self)
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns true if this assembler was instantiated in debug mode.
    pub fn in_debug_mode(&self) -> bool {
        self.in_debug_mode
    }

    /// Returns a reference to the kernel for this assembler.
    ///
    /// If the assembler was instantiated without a kernel, the internal kernel will be empty.
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    // PROGRAM COMPILER
    // --------------------------------------------------------------------------------------------

    /// Compiles the provided source code into a [Program]. The resulting program can be executed
    /// on Miden VM.
    ///
    /// # Errors
    /// Returns an error if parsing or compilation of the specified program fails.
    pub fn compile<S>(&self, source: S) -> Result<Program, AssemblyError>
    where
        S: AsRef<str>,
    {
        // parse the program into an AST
        let source = source.as_ref();
        let program = ProgramAst::parse(source)?;

        // compile the program and return
        self.compile_ast(&program)
    }

    /// Compiles the provided abstract syntax tree into a [Program]. The resulting program can be
    /// executed on Miden VM.
    ///
    /// # Errors
    /// Returns an error if the compilation of the specified program fails.
    #[instrument("compile_ast", skip_all)]
    pub fn compile_ast(&self, program: &ProgramAst) -> Result<Program, AssemblyError> {
        // compile the program
        let mut context = AssemblyContext::for_program(Some(program));
        let program_root = self.compile_in_context(program, &mut context)?;

        // convert the context into a call block table for the program
        let cb_table = context.into_cb_table(&self.proc_cache.borrow())?;

        // build and return the program
        Ok(Program::with_kernel(program_root, self.kernel.clone(), cb_table))
    }

    /// Compiles the provided [ProgramAst] into a program and returns the program root
    /// ([CodeBlock]). Mutates the provided context by adding all of the call targets of
    /// the program to the [CallSet].
    ///
    /// # Errors
    /// - If the provided context is not appropriate for compiling a program.
    /// - If any of the local procedures defined in the program are exported.
    /// - If compilation of any of the local procedures fails.
    /// - if compilation of the program body fails.
    pub fn compile_in_context(
        &self,
        program: &ProgramAst,
        context: &mut AssemblyContext,
    ) -> Result<CodeBlock, AssemblyError> {
        // check to ensure that the context is appropriate for compiling a program
        if context.current_context_name() != ProcedureName::main().as_str() {
            return Err(AssemblyError::InvalidProgramAssemblyContext);
        }

        // compile all local procedures; this will add the procedures to the specified context
        for proc_ast in program.procedures() {
            if proc_ast.is_export {
                return Err(AssemblyError::exported_proc_in_program(&proc_ast.name));
            }
            self.compile_procedure(proc_ast, context)?;
        }

        // compile the program body
        let program_root = self.compile_body(program.body().nodes().iter(), context, None)?;

        Ok(program_root)
    }

    // MODULE COMPILER
    // --------------------------------------------------------------------------------------------

    /// Compiles all procedures in the specified module and adds them to the procedure cache.
    /// Returns a vector of procedure digests for all exported procedures in the module.
    ///
    /// # Errors
    /// - If a module with the same path already exists in the module stack of the
    ///   [AssemblyContext].
    /// - If a lock to the [ProcedureCache] can not be attained.
    #[instrument(level = "trace",
                 name = "compile_module",
                 fields(module = path.unwrap_or(&LibraryPath::anon_path()).path()), skip_all)]
    pub fn compile_module(
        &self,
        module: &ModuleAst,
        path: Option<&LibraryPath>,
        context: &mut AssemblyContext,
    ) -> Result<Vec<RpoDigest>, AssemblyError> {
        // a variable to track MAST roots of all procedures exported from this module
        web_sys::console::log_1(&"compile 1".into());
        let mut proc_roots = Vec::new();
        context.begin_module(path.unwrap_or(&LibraryPath::anon_path()), module)?;
        web_sys::console::log_1(&"compile 2".into());

        // process all re-exported procedures
        for reexporteed_proc in module.reexported_procs().iter() {
            // make sure the re-exported procedure is loaded into the procedure cache
            let ref_proc_id = reexporteed_proc.proc_id();
            web_sys::console::log_1(&"compile 3".into());
            self.ensure_procedure_is_in_cache(&ref_proc_id, context).map_err(|_| {
                web_sys::console::log_1(&"compile 3.1".into());
                AssemblyError::ReExportedProcModuleNotFound(reexporteed_proc.clone())
            })?;
            web_sys::console::log_1(&"compile 4".into());

            // if the library path is provided, build procedure ID for the alias and add it to the
            // procedure cache
            let proc_mast_root = if let Some(path) = path {
                let proc_name = reexporteed_proc.name();
                web_sys::console::log_1(&"compile 5".into());
                let alias_proc_id = ProcedureId::from_name(proc_name, path);
                web_sys::console::log_1(&"compile 6".into());
                self.proc_cache
                    .try_borrow_mut()
                    .map_err(|_| AssemblyError::InvalidCacheLock)?
                    .insert_proc_alias(alias_proc_id, ref_proc_id)?
            } else {
                web_sys::console::log_1(&"compile 7".into());
                // log 
                let thing = self.proc_cache
                    .try_borrow_mut()
                    .map_err(|_| AssemblyError::InvalidCacheLock)?
                    .get_proc_root_by_id(&ref_proc_id);
                web_sys::console::log_1(&format!("compile 7 {:?}", thing).into());

                self.proc_cache
                    .try_borrow_mut()
                    .map_err(|_| AssemblyError::InvalidCacheLock)?
                    .get_proc_root_by_id(&ref_proc_id)
                    .expect("procedure ID not in cache")
            };

            // add the MAST root of the re-exported procedure to the set of procedures exported
            // from this module
            proc_roots.push(proc_mast_root);
        }

        // compile all local (internal end exported) procedures in the module; once the compilation
        // is complete, we get all compiled procedures (and their combined callset) from the
        // context
        web_sys::console::log_1(&"compile 8".into());
        for proc_ast in module.procs().iter() {
            self.compile_procedure(proc_ast, context)?;
        }
        web_sys::console::log_1(&"compile 9".into());
        let (module_procs, module_callset) = context.complete_module()?;

        // add the compiled procedures to the assembler's cache. the procedures are added to the
        // cache only if:
        // - a procedure is exported from the module, or
        // - a procedure is present in the combined callset - i.e., it is an internal procedure
        //   which has been invoked via a local call instruction.
        web_sys::console::log_1(&"compile 10".into());
        for (proc_index, proc) in module_procs.into_iter().enumerate() {
            web_sys::console::log_1(&"compile 101".into());
            if proc.is_export() {
                web_sys::console::log_1(&"compile 102".into());
                proc_roots.push(proc.mast_root());
            }
            web_sys::console::log_1(&"compile 11".into());

            if proc.is_export() || module_callset.contains(&proc.mast_root()) {
                // build the procedure ID if this module has the library path
                let proc_id = build_procedure_id(path, &proc, proc_index);
                web_sys::console::log_1(&"compile 111".into());

                // this is safe because we fail if the cache is borrowed.
                self.proc_cache
                    .try_borrow_mut()
                    .map_err(|_| AssemblyError::InvalidCacheLock)?
                    .insert(proc, proc_id)?;

                web_sys::console::log_1(&"compile 112".into());
            }
            web_sys::console::log_1(&"compile 12".into());
        }
        web_sys::console::log_1(&"compile 13".into());
        // console log where the kernel on the context is empty
        web_sys::console::log_1(&format!("compile 13 {:?}", context.kernel_is_some()).into());
        // console log the proc roots
        for proc_root in &proc_roots {
            web_sys::console::log_1(&format!("{:?}", proc_root).into());
        }

        Ok(proc_roots)
    }

    // PROCEDURE COMPILER
    // --------------------------------------------------------------------------------------------

    /// Compiles procedure AST into MAST and adds the complied procedure to the provided context.
    fn compile_procedure(
        &self,
        proc: &ProcedureAst,
        context: &mut AssemblyContext,
    ) -> Result<(), AssemblyError> {
        context.begin_proc(&proc.name, proc.is_export, proc.num_locals)?;
        let code = if proc.num_locals > 0 {
            // for procedures with locals, we need to update fmp register before and after the
            // procedure body is executed. specifically:
            // - to allocate procedure locals we need to increment fmp by the number of locals
            // - to deallocate procedure locals we need to decrement it by the same amount
            let num_locals = Felt::from(proc.num_locals);
            let wrapper = BodyWrapper {
                prologue: vec![Operation::Push(num_locals), Operation::FmpUpdate],
                epilogue: vec![Operation::Push(-num_locals), Operation::FmpUpdate],
            };
            self.compile_body(proc.body.nodes().iter(), context, Some(wrapper))?
        } else {
            self.compile_body(proc.body.nodes().iter(), context, None)?
        };

        context.complete_proc(code);

        Ok(())
    }

    // CODE BODY COMPILER
    // --------------------------------------------------------------------------------------------

    /// TODO: add comments
    fn compile_body<A, N>(
        &self,
        body: A,
        context: &mut AssemblyContext,
        wrapper: Option<BodyWrapper>,
    ) -> Result<CodeBlock, AssemblyError>
    where
        A: Iterator<Item = N>,
        N: Borrow<Node>,
    {
        let mut blocks: Vec<CodeBlock> = Vec::new();
        let mut span = SpanBuilder::new(wrapper);

        for node in body {
            match node.borrow() {
                Node::Instruction(inner) => {
                    if let Some(block) = self.compile_instruction(inner, &mut span, context)? {
                        span.extract_span_into(&mut blocks);
                        blocks.push(block);
                    }
                }

                Node::IfElse {
                    true_case,
                    false_case,
                } => {
                    span.extract_span_into(&mut blocks);

                    let true_case = self.compile_body(true_case.nodes().iter(), context, None)?;

                    // else is an exception because it is optional; hence, will have to be replaced
                    // by noop span
                    let false_case = if !false_case.nodes().is_empty() {
                        self.compile_body(false_case.nodes().iter(), context, None)?
                    } else {
                        CodeBlock::new_span(vec![Operation::Noop])
                    };

                    let block = CodeBlock::new_split(true_case, false_case);

                    blocks.push(block);
                }

                Node::Repeat { times, body } => {
                    span.extract_span_into(&mut blocks);

                    let block = self.compile_body(body.nodes().iter(), context, None)?;

                    for _ in 0..*times {
                        blocks.push(block.clone());
                    }
                }

                Node::While { body } => {
                    span.extract_span_into(&mut blocks);

                    let block = self.compile_body(body.nodes().iter(), context, None)?;
                    let block = CodeBlock::new_loop(block);

                    blocks.push(block);
                }
            }
        }

        span.extract_final_span_into(&mut blocks);
        Ok(if blocks.is_empty() {
            CodeBlock::new_span(vec![Operation::Noop])
        } else {
            combine_blocks(blocks)
        })
    }

    // PROCEDURE CACHE
    // --------------------------------------------------------------------------------------------

    /// Ensures that a procedure with the specified [ProcedureId] exists in the cache. Otherwise,
    /// attempt to fetch it from the module provider, compile, and check again.
    ///
    /// If `Ok` is returned, the procedure can be safely unwrapped from the cache.
    ///
    /// # Panics
    ///
    /// This function will panic if the internal procedure cache is mutably borrowed somewhere.
    fn ensure_procedure_is_in_cache(
        &self,
        proc_id: &ProcedureId,
        context: &mut AssemblyContext,
    ) -> Result<(), AssemblyError> {
        web_sys::console::log_1(&"ensure 0".into());
        if !self.proc_cache.borrow().contains_id(proc_id) {
            // if procedure is not in cache, try to get its module and compile it
            let module = self.module_provider.get_module(proc_id).ok_or_else(|| {
                let proc_name = context.get_imported_procedure_name(proc_id);
                AssemblyError::imported_proc_module_not_found(proc_id, proc_name)
            })?;
            web_sys::console::log_1(&"ensure 1".into());
            self.compile_module(&module.ast, Some(&module.path), context)?;
            web_sys::console::log_1(&"ensure 2".into());
            // if the procedure is still not in cache, then there was some error
            // log if the proc_id is in the cache
            web_sys::console::log_1(&format!("ensure {:?}", self.proc_cache.borrow().contains_id(proc_id)).into());
            if !self.proc_cache.borrow().contains_id(proc_id) {
                return Err(AssemblyError::imported_proc_not_found_in_module(
                    proc_id,
                    &module.path,
                ));
            }
            web_sys::console::log_1(&"ensure 3".into());
        }

        Ok(())
    }

    // CODE BLOCK BUILDER
    // --------------------------------------------------------------------------------------------
    /// Returns the [CodeBlockTable] associated with the [AssemblyContext].
    ///
    /// # Errors
    /// Returns an error if a required procedure is not found in the [Assembler] procedure cache.
    pub fn build_cb_table(
        &self,
        context: AssemblyContext,
    ) -> Result<CodeBlockTable, AssemblyError> {
        context.into_cb_table(&self.proc_cache.borrow())
    }
}

// BODY WRAPPER
// ================================================================================================

/// Contains a set of operations which need to be executed before and after a sequence of AST
/// nodes (i.e., code body).
struct BodyWrapper {
    prologue: Vec<Operation>,
    epilogue: Vec<Operation>,
}

// HELPER FUNCTIONS
// ================================================================================================
fn combine_blocks_new(blocks: Vec<CodeBlock>) -> CodeBlock {
    debug_assert!(!blocks.is_empty(), "cannot combine empty block list");

    // In-place merging of spans
    let mut merged_blocks = Vec::new();
    let mut span_start = 0; 

    for (i, block) in blocks.into_iter().enumerate() {
        if block.is_span() {
            // If we've accumulated spans, combine them before pushing
            if span_start != i {  
                let combined_span = combine_spans(&mut merged_blocks[span_start..i].to_vec());
                merged_blocks.push(combined_span);
                span_start = i + 1; // Reset span start
            }
        } else {
            // Non-span: If we had spans, combine; then push the current block
            if span_start != i {
                let combined_span = combine_spans(&mut merged_blocks[span_start..i].to_vec());
                merged_blocks.push(combined_span);
            }
            merged_blocks.push(block);
            span_start = i + 1; // Reset span start
        }
    }

    // Handle leftover spans
    if span_start < merged_blocks.len() {
        let combined_span = combine_spans(&mut merged_blocks[span_start..].to_vec());
        merged_blocks.push(combined_span);
    }

    // Build the tree (using indices to avoid excessive copying)
    let mut blocks = merged_blocks;
    while blocks.len() > 1 {
        let last_index = if blocks.len() % 2 == 0 { blocks.len() - 2 } else { blocks.len() - 1 };

        for i in (0..blocks.len() / 2).rev() {
            let pair = (2 * i, 2 * i + 1);
            blocks[i] = CodeBlock::new_join((blocks[pair.0].clone(), blocks[pair.1].clone()).into());
        }

        blocks.truncate(blocks.len() / 2 + (blocks.len() % 2)); // Shrink, considering odd lengths
    }

    debug_assert!(!blocks.is_empty(), "no blocks");
    blocks.remove(0)
}

fn combine_blocks(mut blocks: Vec<CodeBlock>) -> CodeBlock {
    debug_assert!(!blocks.is_empty(), "cannot combine empty block list");
    // merge consecutive Span blocks.
    let mut merged_blocks: Vec<CodeBlock> = Vec::with_capacity(blocks.len());
    // Keep track of all the consecutive Span blocks and are merged together when
    // there is a discontinuity.
    let mut contiguous_spans: Vec<CodeBlock> = Vec::new();

    blocks.drain(0..).for_each(|block| {
        if block.is_span() {
            contiguous_spans.push(block);
        } else {
            if !contiguous_spans.is_empty() {
                merged_blocks.push(combine_spans(&mut contiguous_spans));
            }
            merged_blocks.push(block);
        }
    });
    if !contiguous_spans.is_empty() {
        merged_blocks.push(combine_spans(&mut contiguous_spans));
    }

    // build a binary tree of blocks joining them using JOIN blocks
    let mut blocks = merged_blocks;
    while blocks.len() > 1 {
        let last_block = if blocks.len() % 2 == 0 { None } else { blocks.pop() };

        let mut grouped_blocks = Vec::new();
        core::mem::swap(&mut blocks, &mut grouped_blocks);
        let mut grouped_blocks = group_vector_elements::<CodeBlock, 2>(grouped_blocks);
        grouped_blocks.drain(0..).for_each(|pair| {
            blocks.push(CodeBlock::new_join(pair)); // not this
        });

        if let Some(block) = last_block {
            blocks.push(block);
        }
    }

    debug_assert!(!blocks.is_empty(), "no blocks");
    // console log the length of blocks
    web_sys::console::log_1(&format!("blocks length: {:?}", blocks.len()).into());
    blocks.remove(0)
}

/// Combines a vector of SPAN blocks into a single SPAN block.
///
/// # Panics
/// Panics if any of the provided blocks is not a SPAN block.
fn combine_spans(spans: &mut Vec<CodeBlock>) -> CodeBlock {
    if spans.len() == 1 {
        return spans.remove(0);
    }

    let mut ops = Vec::<Operation>::new();
    let mut decorators = DecoratorList::new();
    spans.drain(0..).for_each(|block| {
        if let CodeBlock::Span(span) = block {
            for decorator in span.decorators() {
                decorators.push((decorator.0 + ops.len(), decorator.1.clone()));
            }
            for batch in span.op_batches() {
                ops.extend_from_slice(batch.ops());
            }
        } else {
            panic!("CodeBlock was expected to be a Span Block, got {block:?}.");
        }
    });
    CodeBlock::new_span_with_decorators(ops, decorators)
}

/// Builds a procedure ID based on the provided parameters.
///
/// Returns [ProcedureId] if `path` is provided, [None] otherwise.
fn build_procedure_id(
    path: Option<&LibraryPath>,
    proc: &NamedProcedure,
    proc_index: usize,
) -> Option<ProcedureId> {
    web_sys::console::log_1(&"proc id 1".into());
    let mut proc_id = None;
    if let Some(path) = path {
        // print the library path
        web_sys::console::log_1(&path.path().into());
        web_sys::console::log_1(&"proc id 2".into());
        if proc.is_export() {
            proc_id = Some(ProcedureId::from_name(proc.name(), path));
        } else {
            proc_id = Some(ProcedureId::from_index(proc_index as u16, path))
        }
    }
    web_sys::console::log_1(&"proc id 3".into());
    proc_id
}
