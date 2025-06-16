	.file	"ggml-bitnet-mad.cpp"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	quantize_i2_s
	.type	quantize_i2_s, @function
quantize_i2_s:
.LFB1932:
	.cfi_startproc
	addi	sp,sp,-96
	.cfi_def_cfa_offset 96
	sd	s5,40(sp)
	sd	s7,24(sp)
	.cfi_offset 21, -56
	.cfi_offset 23, -72
	mv	s5,a2
	mv	s7,a3
	mul	s7,s7,s5
	sd	s0,80(sp)
	sd	s1,72(sp)
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	mv	s0,a0
	mv	s1,a1
	li	a0,36
	mv	a1,a3
	sd	s6,32(sp)
	sd	ra,88(sp)
	sd	s3,56(sp)
	fsd	fs0,8(sp)
	.cfi_offset 22, -64
	.cfi_offset 1, -8
	.cfi_offset 19, -40
	.cfi_offset 40, -88
	call	ggml_row_size@plt
	mv	s6,a0
	ble	s7,zero,.L2
	fmv.s.x	fs0,zero
	sd	s4,48(sp)
	.cfi_offset 20, -48
	slli	s4,s7,2
	sd	s2,64(sp)
	.cfi_offset 18, -32
	add	s4,s0,s4
	mv	a5,s0
.L5:
	flw	fa5,0(a5)
	addi	a5,a5,4
	fabs.s	fa5,fa5
	fgt.s	a4,fa5,fs0
	bne	a4,zero,.L25
	bne	s4,a5,.L5
.L26:
	fmv.s.x	fa5,zero
	feq.s	a5,fs0,fa5
	beq	a5,zero,.L8
	flw	fs0,.LC0,a5
.L8:
	mv	a0,s7
	call	malloc@plt
	fmv.s.x	fa2,zero
	mv	s3,a0
	mv	s2,a0
	mv	a5,a0
	flw	fa3,.LC1,a4
	li	a3,1
.L12:
	flw	fa5,0(s0)
	fabs.s	fa4,fa5
	flt.s	a4,fa4,fa3
	bne	a4,zero,.L9
	fmul.s	fa5,fa5,fs0
	addi	a5,a5,1
	addi	s0,s0,4
	fgt.s	a4,fa5,fa2
	slliw	a4,a4,1
	sb	a4,-1(a5)
	bne	s4,s0,.L12
.L11:
	mul	s5,s5,s6
	li	a1,0
	mv	a0,s1
	mv	a2,s5
	call	memset@plt
	li	a5,127
	srai	t3,s7,7
	li	t1,0
	li	a7,128
	ble	s7,a5,.L23
.L13:
	slli	a6,t1,5
	mv	a0,s2
	li	a4,0
.L15:
	sraiw	a5,a4,5
	slliw	a5,a5,3
	srai	a3,a4,2
	add	a5,a5,a6
	add	a5,a5,a3
	lbu	a2,0(a0)
	add	a5,s1,a5
	lbu	a3,0(a5)
	andi	a1,a4,3
	slliw	a1,a1,1
	sllw	a2,a2,a1
	or	a3,a3,a2
	sb	a3,0(a5)
	addiw	a4,a4,1
	addi	a0,a0,1
	bne	a4,a7,.L15
	addi	t1,t1,1
	addi	s2,s2,128
	blt	t1,t3,.L13
.L23:
	ld	s2,64(sp)
	.cfi_restore 18
	ld	s4,48(sp)
	.cfi_restore 20
.L7:
	add	s1,s1,s5
	fsw	fs0,-4(s1)
	mv	a0,s3
	call	free@plt
	ld	ra,88(sp)
	.cfi_restore 1
	ld	s0,80(sp)
	.cfi_restore 8
	ld	s1,72(sp)
	.cfi_restore 9
	ld	s3,56(sp)
	.cfi_restore 19
	ld	s6,32(sp)
	.cfi_restore 22
	ld	s7,24(sp)
	.cfi_restore 23
	fld	fs0,8(sp)
	.cfi_restore 40
	mv	a0,s5
	ld	s5,40(sp)
	.cfi_restore 21
	addi	sp,sp,96
	.cfi_def_cfa_offset 0
	jr	ra
.L25:
	.cfi_def_cfa_offset 96
	.cfi_offset 1, -8
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	.cfi_offset 40, -88
	fmv.s	fs0,fa5
	bne	s4,a5,.L5
	j	.L26
.L9:
	sb	a3,0(a5)
	addi	s0,s0,4
	addi	a5,a5,1
	bne	s4,s0,.L12
	j	.L11
.L2:
	.cfi_restore 18
	.cfi_restore 20
	mul	s5,s5,s6
	mv	a0,s7
	call	malloc@plt
	mv	s3,a0
	li	a1,0
	mv	a0,s1
	mv	a2,s5
	call	memset@plt
	flw	fs0,.LC0,a5
	j	.L7
	.cfi_endproc
.LFE1932:
	.size	quantize_i2_s, .-quantize_i2_s
	.align	1
	.globl	ggml_vec_dot_i2_i8_s
	.type	ggml_vec_dot_i2_i8_s, @function
ggml_vec_dot_i2_i8_s:
.LFB1933:
	.cfi_startproc
	ble	a0,zero,.L30
	li	a2,0
	li	a7,0
.L29:
	sraiw	a4,a2,2
	add	a4,a3,a4
	not	a6,a2
	lbu	a4,0(a4)
	andi	a6,a6,3
	add	t1,a5,a2
	slliw	a6,a6,1
	lb	t1,0(t1)
	sraw	a4,a4,a6
	andi	a4,a4,3
	addiw	a4,a4,-1
	mulw	a4,a4,t1
	addi	a2,a2,1
	addw	a7,a4,a7
	bne	a0,a2,.L29
	fcvt.s.w	fa5,a7
	fsw	fa5,0(a1)
	ret
.L30:
	fmv.s.x	fa5,zero
	fsw	fa5,0(a1)
	ret
	.cfi_endproc
.LFE1933:
	.size	ggml_vec_dot_i2_i8_s, .-ggml_vec_dot_i2_i8_s
	.section	.rodata.cst4,"aM",@progbits,4
	.align	2
.LC0:
	.word	1065353216
	.align	2
.LC1:
	.word	897988541
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
