	.file	"ggml-bitnet-lut.cpp"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	ggml_bitnet_init
	.type	ggml_bitnet_init, @function
ggml_bitnet_init:
.LFB1596:
	.cfi_startproc
	addi	sp,sp,-16
	.cfi_def_cfa_offset 16
	sd	s0,0(sp)
	.cfi_offset 8, -16
	lla	s0,.LANCHOR0
	lbu	a5,0(s0)
	sd	ra,8(sp)
	.cfi_offset 1, -8
	bne	a5,zero,.L1
	ld	a5,8(s0)
	li	a4,1
	sb	a4,0(s0)
	beq	a5,zero,.L7
.L1:
	ld	ra,8(sp)
	.cfi_remember_state
	.cfi_restore 1
	ld	s0,0(sp)
	.cfi_restore 8
	addi	sp,sp,16
	.cfi_def_cfa_offset 0
	jr	ra
.L7:
	.cfi_restore_state
	li	a0,262144
	call	_Znam@plt
	ld	ra,8(sp)
	.cfi_restore 1
	sd	a0,8(s0)
	ld	s0,0(sp)
	.cfi_restore 8
	addi	sp,sp,16
	.cfi_def_cfa_offset 0
	jr	ra
	.cfi_endproc
.LFE1596:
	.size	ggml_bitnet_init, .-ggml_bitnet_init
	.align	1
	.globl	ggml_bitnet_free
	.type	ggml_bitnet_free, @function
ggml_bitnet_free:
.LFB1597:
	.cfi_startproc
	lla	a5,.LANCHOR0
	lbu	a4,0(a5)
	beq	a4,zero,.L18
	ld	a0,8(a5)
	sb	zero,.LANCHOR0,a5
	beq	a0,zero,.L19
	addi	sp,sp,-16
	.cfi_def_cfa_offset 16
	sd	ra,8(sp)
	.cfi_offset 1, -8
	call	_ZdaPv@plt
	ld	ra,8(sp)
	.cfi_restore 1
	sd	zero,.LANCHOR0+8,a5
	addi	sp,sp,16
	.cfi_def_cfa_offset 0
	jr	ra
.L19:
	sd	zero,.LANCHOR0+8,a5
	ret
.L18:
	ret
	.cfi_endproc
.LFE1597:
	.size	ggml_bitnet_free, .-ggml_bitnet_free
	.align	1
	.globl	ggml_bitnet_can_mul_mat
	.type	ggml_bitnet_can_mul_mat, @function
ggml_bitnet_can_mul_mat:
.LFB1599:
	.cfi_startproc
	lw	a5,0(a0)
	li	a4,38
	bgtu	a5,a4,.L24
	li	a0,1
	slli	a0,a0,38
	addi	a0,a0,271
	srl	a0,a0,a5
	andi	a0,a0,1
	beq	a0,zero,.L23
	lw	a5,0(a1)
	li	a0,0
	bne	a5,zero,.L23
	lw	a5,0(a2)
	bne	a5,zero,.L23
	ld	a0,24(a1)
	slti	a0,a0,2
	ret
.L24:
	li	a0,0
.L23:
	ret
	.cfi_endproc
.LFE1599:
	.size	ggml_bitnet_can_mul_mat, .-ggml_bitnet_can_mul_mat
	.align	1
	.globl	ggml_bitnet_mul_mat_get_wsize
	.type	ggml_bitnet_mul_mat_get_wsize, @function
ggml_bitnet_mul_mat_get_wsize:
.LFB1600:
	.cfi_startproc
	ld	a5,16(a1)
	ld	a4,24(a1)
	slli	a0,a5,4
	sub	a0,a0,a5
	addi	a0,a0,8
	mul	a0,a0,a4
	addi	a0,a0,-1
	andi	a0,a0,-64
	addi	a0,a0,64
	ret
	.cfi_endproc
.LFE1600:
	.size	ggml_bitnet_mul_mat_get_wsize, .-ggml_bitnet_mul_mat_get_wsize
	.align	1
	.globl	ggml_bitnet_get_type_bits
	.type	ggml_bitnet_get_type_bits, @function
ggml_bitnet_get_type_bits:
.LFB1602:
	.cfi_startproc
	li	a5,2
	beq	a0,a5,.L33
	addi	a0,a0,-38
	seqz	a0,a0
	slli	a0,a0,1
	ret
.L33:
	li	a0,4
	ret
	.cfi_endproc
.LFE1602:
	.size	ggml_bitnet_get_type_bits, .-ggml_bitnet_get_type_bits
	.align	1
	.globl	_Z19ggml_bitnet_mul_matP11ggml_tensorPKS_S2_PKvm
	.type	_Z19ggml_bitnet_mul_matP11ggml_tensorPKS_S2_PKvm, @function
_Z19ggml_bitnet_mul_matP11ggml_tensorPKS_S2_PKvm:
.LFB1603:
	.cfi_startproc
	addi	sp,sp,-64
	.cfi_def_cfa_offset 64
	sd	s3,24(sp)
	sd	s5,8(sp)
	.cfi_offset 19, -40
	.cfi_offset 21, -56
	lw	s3,24(a1)
	ld	s5,16(a2)
	mv	a5,a2
	sd	s2,32(sp)
	mulw	a2,s5,s3
	.cfi_offset 18, -32
	ld	s2,16(a1)
	sd	s0,48(sp)
	sd	s1,40(sp)
	sd	s4,16(sp)
	sd	s6,0(sp)
	sd	ra,56(sp)
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 20, -48
	.cfi_offset 22, -64
	.cfi_offset 1, -8
	ld	s1,256(a5)
	ld	s4,256(a1)
	ld	a5,256(a0)
	sext.w	a4,a2
	sext.w	s0,s2
	sext.w	s6,s5
	ble	a4,zero,.L35
	slli	a4,a2,32
	mv	a0,a5
	srli	a2,a4,30
	li	a1,0
	call	memset@plt
	mv	a5,a0
.L35:
	ble	s3,zero,.L34
	ble	s6,zero,.L34
	addiw	a4,s5,-1
	slli	a3,a4,32
	addiw	t3,s2,-1
	srli	a4,a3,30
	addi	a5,a5,4
	li	t5,-4
	slli	t3,t3,32
	slli	t4,s6,2
	add	a6,a5,a4
	li	a7,0
	li	t1,0
	addi	t6,s4,4
	sub	t5,t5,a4
	srli	t3,t3,32
.L37:
	add	a3,t3,a7
	slli	a0,a7,2
	slli	a3,a3,2
	add	a2,t5,a6
	add	a0,s4,a0
	add	a3,t6,a3
	li	a1,0
.L42:
	slli	a4,a1,2
	fmv.s.x	fa5,zero
	add	a4,s1,a4
	mv	a5,a0
	ble	s0,zero,.L41
.L38:
	flw	fa3,0(a5)
	flw	fa4,0(a4)
	addi	a5,a5,4
	addi	a4,a4,4
	fmadd.s	fa5,fa3,fa4,fa5
	bne	a5,a3,.L38
.L41:
	fsw	fa5,0(a2)
	addi	a2,a2,4
	addw	a1,s0,a1
	bne	a6,a2,.L42
	addiw	t1,t1,1
	add	a6,a6,t4
	addw	a7,s0,a7
	bne	s3,t1,.L37
.L34:
	ld	ra,56(sp)
	.cfi_restore 1
	ld	s0,48(sp)
	.cfi_restore 8
	ld	s1,40(sp)
	.cfi_restore 9
	ld	s2,32(sp)
	.cfi_restore 18
	ld	s3,24(sp)
	.cfi_restore 19
	ld	s4,16(sp)
	.cfi_restore 20
	ld	s5,8(sp)
	.cfi_restore 21
	ld	s6,0(sp)
	.cfi_restore 22
	addi	sp,sp,64
	.cfi_def_cfa_offset 0
	jr	ra
	.cfi_endproc
.LFE1603:
	.size	_Z19ggml_bitnet_mul_matP11ggml_tensorPKS_S2_PKvm, .-_Z19ggml_bitnet_mul_matP11ggml_tensorPKS_S2_PKvm
	.bss
	.align	3
	.set	.LANCHOR0,. + 0
	.type	_ZL11initialized, @object
	.size	_ZL11initialized, 1
_ZL11initialized:
	.zero	1
	.zero	7
	.type	_ZL20bitnet_tensor_extras, @object
	.size	_ZL20bitnet_tensor_extras, 8
_ZL20bitnet_tensor_extras:
	.zero	8
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
